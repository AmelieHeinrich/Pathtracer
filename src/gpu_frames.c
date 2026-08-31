#include "gpu_internal.h"

#include <GLFW/glfw3.h>

#include <string.h>

// ---------------------------------------------------------------------------
// command helpers
// ---------------------------------------------------------------------------

void gpu_cmd_image_barrier(VkCommandBuffer cmd, const gpu_image_barrier_t *barrier)
{
    const VkImageMemoryBarrier2 image_barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = barrier->src_stage,
        .srcAccessMask = barrier->src_access,
        .dstStageMask = barrier->dst_stage,
        .dstAccessMask = barrier->dst_access,
        .oldLayout = barrier->old_layout,
        .newLayout = barrier->new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = barrier->image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        },
    };

    const VkDependencyInfo dependency = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &image_barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dependency);
}

void gpu_cmd_begin_rendering(VkCommandBuffer cmd, VkImageView view, VkExtent2D extent,
                             const VkClearColorValue *clear)
{
    const VkRenderingAttachmentInfo color = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = view,
        // Legal as an attachment layout because unifiedImageLayouts is enabled, and just
        // as fast as COLOR_ATTACHMENT_OPTIMAL would be.
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = {.color = clear ? *clear : (VkClearColorValue){{0}}},
    };

    const VkRenderingInfo info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {.extent = extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color,
    };
    vkCmdBeginRendering(cmd, &info);
}

void gpu_cmd_end_rendering(VkCommandBuffer cmd)
{
    vkCmdEndRendering(cmd);
}

// ---------------------------------------------------------------------------
// frame ring
// ---------------------------------------------------------------------------

void gpu_frames_init(gpu_frames_t *frames, gpu_device_t *device)
{
    memset(frames, 0, sizeof(*frames));
    frames->device = device;

    // A timeline semaphore replaces the per-frame fences of the older idiom: one object
    // paces the whole ring, and frame N is complete once it reaches value N + 1.
    VkSemaphoreTypeCreateInfo type_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0,
    };
    const VkSemaphoreCreateInfo timeline_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &type_info,
    };
    VK_CHECK(vkCreateSemaphore(device->device, &timeline_info, NULL, &frames->timeline));

    for (uint32_t i = 0; i < GPU_FRAMES_IN_FLIGHT; ++i) {
        gpu_frame_t *frame = &frames->frame[i];
        frame->slot = i;

        const VkCommandPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            // The pool is reset wholesale every frame rather than per buffer.
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = device->universal_family,
        };
        VK_CHECK(vkCreateCommandPool(device->device, &pool_info, NULL, &frame->pool));

        const VkCommandBufferAllocateInfo alloc_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = frame->pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VK_CHECK(vkAllocateCommandBuffers(device->device, &alloc_info, &frame->cmd));

        // Binary, because vkAcquireNextImageKHR cannot signal a timeline semaphore.
        const VkSemaphoreCreateInfo semaphore_info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };
        VK_CHECK(vkCreateSemaphore(device->device, &semaphore_info, NULL, &frame->acquire));
    }
}

void gpu_frames_free(gpu_frames_t *frames)
{
    if (!frames->device) {
        return;
    }
    VkDevice device = frames->device->device;

    for (uint32_t i = 0; i < GPU_FRAMES_IN_FLIGHT; ++i) {
        gpu_frame_t *frame = &frames->frame[i];
        vkDestroySemaphore(device, frame->acquire, NULL);
        // Frees the command buffer allocated from it too.
        vkDestroyCommandPool(device, frame->pool, NULL);
    }
    vkDestroySemaphore(device, frames->timeline, NULL);
    memset(frames, 0, sizeof(*frames));
}

// Blocks until the submission that last used this frame slot has completed. Frame N
// signals value N + 1, so the slot's previous occupant was frame N - GPU_FRAMES_IN_FLIGHT.
static void wait_for_slot(gpu_frames_t *frames)
{
    if (frames->submitted < GPU_FRAMES_IN_FLIGHT) {
        return; // the ring has not wrapped yet, nothing has used this slot
    }
    const uint64_t target = frames->submitted - GPU_FRAMES_IN_FLIGHT + 1;

    const VkSemaphoreWaitInfo wait = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &frames->timeline,
        .pValues = &target,
    };
    VK_CHECK(vkWaitSemaphores(frames->device->device, &wait, UINT64_MAX));
}

gpu_frame_t *gpu_frame_begin(gpu_frames_t *frames, gpu_swapchain_t *swapchain)
{
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(swapchain->window, &width, &height);

    const bool resized = swapchain->window_extent.width != (uint32_t)width ||
                         swapchain->window_extent.height != (uint32_t)height;
    if (swapchain->out_of_date || resized || !swapchain->handle) {
        gpu_swapchain_recreate(swapchain);
    }
    if (!swapchain->handle) {
        return NULL; // minimised
    }

    gpu_frame_t *frame = &frames->frame[frames->submitted % GPU_FRAMES_IN_FLIGHT];

    // Only now are this slot's command pool and acquire semaphore free to reuse.
    wait_for_slot(frames);

    const VkResult acquired = vkAcquireNextImageKHR(frames->device->device, swapchain->handle,
                                                    UINT64_MAX, frame->acquire, VK_NULL_HANDLE,
                                                    &frame->image_index);
    if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
        // The spec leaves the semaphore unsignalled when acquire fails, so nothing dangles.
        swapchain->out_of_date = true;
        return NULL;
    }
    if (acquired == VK_SUBOPTIMAL_KHR) {
        // Still a usable image; rebuild before the next frame.
        swapchain->out_of_date = true;
    } else if (acquired != VK_SUCCESS) {
        gpu_fatal("vkAcquireNextImageKHR failed: %s", vk_result_string(acquired));
    }

    frame->image = swapchain->images[frame->image_index];
    frame->view = swapchain->views[frame->image_index];
    frame->extent = swapchain->extent;

    VK_CHECK(vkResetCommandPool(frames->device->device, frame->pool, 0));

    const VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(frame->cmd, &begin_info));

    // UNDEFINED discards the previous contents, which some drivers optimise for. Under
    // unified image layouts GENERAL is the working layout for everything that follows.
    //
    // The destination scope is deliberately ALL_COMMANDS rather than a specific stage:
    // this layer does not know whether the caller will render into the image, blit into
    // it, or write it from a compute shader, and naming only one of those silently
    // fails to order the transition against the others.
    gpu_cmd_image_barrier(frame->cmd, &(gpu_image_barrier_t){
        .image = frame->image,
        .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
        .new_layout = VK_IMAGE_LAYOUT_GENERAL,
        .src_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .src_access = VK_ACCESS_2_NONE,
        .dst_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dst_access = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
    });

    return frame;
}

void gpu_frame_end(gpu_frames_t *frames, gpu_swapchain_t *swapchain, gpu_frame_t *frame)
{
    gpu_device_t *gpu = frames->device;

    // The one transition VK_KHR_unified_image_layouts does not remove: interaction with
    // the present engine still requires PRESENT_SRC_KHR.
    gpu_cmd_image_barrier(frame->cmd, &(gpu_image_barrier_t){
        .image = frame->image,
        .old_layout = VK_IMAGE_LAYOUT_GENERAL,
        .new_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        // ALL_COMMANDS for the same reason as the acquire barrier: whatever the caller
        // last did to this image -- draw, blit, compute write -- must complete first.
        .src_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .src_access = VK_ACCESS_2_MEMORY_WRITE_BIT,
        // The semaphore synchronises the presentation engine's *read*, but something
        // still has to order this transition's write ahead of the semaphore signal.
        // A NONE destination scope orders nothing, so match the signal's stage below.
        .dst_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dst_access = VK_ACCESS_2_NONE,
    });
    VK_CHECK(vkEndCommandBuffer(frame->cmd));

    const VkSemaphoreSubmitInfo wait = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = frame->acquire,
        .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
    };
    const VkSemaphoreSubmitInfo signals[] = {
        {
            // Binary, and indexed per image: a present may still be pending on it.
            // Signalled from ALL_COMMANDS so the signal is ordered after the
            // PRESENT_SRC_KHR transition, not just after colour writes.
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = swapchain->release[frame->image_index],
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        },
        {
            // Timeline, for CPU-side pacing only.
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = frames->timeline,
            .value = frames->submitted + 1,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        },
    };
    const VkCommandBufferSubmitInfo cmd_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = frame->cmd,
    };
    const VkSubmitInfo2 submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &wait,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmd_info,
        .signalSemaphoreInfoCount = PT_COUNT(signals),
        .pSignalSemaphoreInfos = signals,
    };
    VK_CHECK(vkQueueSubmit2(gpu->universal_queue, 1, &submit, VK_NULL_HANDLE));

    const VkPresentInfoKHR present = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &swapchain->release[frame->image_index],
        .swapchainCount = 1,
        .pSwapchains = &swapchain->handle,
        .pImageIndices = &frame->image_index,
    };
    const VkResult presented = vkQueuePresentKHR(gpu->universal_queue, &present);
    if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) {
        swapchain->out_of_date = true;
    } else if (presented != VK_SUCCESS) {
        gpu_fatal("vkQueuePresentKHR failed: %s", vk_result_string(presented));
    }

    // Counted even when the present was stale: the submit happened and signalled.
    frames->submitted++;
}
