#include "gpu_internal.h"

#include <string.h>

// Picks a memory type satisfying `type_bits` from the device, preferring `preferred`
// properties and falling back to `required`.
static uint32_t find_memory_type(gpu_device_t *device, uint32_t type_bits,
                                 VkMemoryPropertyFlags required,
                                 VkMemoryPropertyFlags preferred)
{
    const VkPhysicalDeviceMemoryProperties *props = &device->memory_properties;

    for (uint32_t i = 0; i < props->memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags flags = props->memoryTypes[i].propertyFlags;
        if ((type_bits & (1u << i)) && (flags & preferred) == preferred) {
            return i;
        }
    }
    for (uint32_t i = 0; i < props->memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags flags = props->memoryTypes[i].propertyFlags;
        if ((type_bits & (1u << i)) && (flags & required) == required) {
            return i;
        }
    }

    gpu_fatal("no memory type matching bits 0x%x with properties 0x%x", type_bits, required);
    return 0;
}

static void memory_kind_flags(gpu_memory_kind_t kind, VkMemoryPropertyFlags *required,
                              VkMemoryPropertyFlags *preferred)
{
    switch (kind) {
    case GPU_MEMORY_UPLOAD:
        *required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        // Device-local host-visible memory (ReBAR) when the driver exposes it.
        *preferred = *required | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case GPU_MEMORY_READBACK:
        *required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        *preferred = *required | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        break;
    case GPU_MEMORY_DEVICE:
    default:
        *required = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        *preferred = *required;
        break;
    }
}

gpu_buffer_t gpu_buffer_create(gpu_device_t *device, VkDeviceSize size,
                               VkBufferUsageFlags usage, gpu_memory_kind_t kind)
{
    gpu_buffer_t buffer = {.size = size};

    const VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VK_CHECK(vkCreateBuffer(device->device, &buffer_info, NULL, &buffer.handle));

    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device->device, buffer.handle, &requirements);

    VkMemoryPropertyFlags required = 0;
    VkMemoryPropertyFlags preferred = 0;
    memory_kind_flags(kind, &required, &preferred);

    // Without this flag vkGetBufferDeviceAddress on the buffer is invalid usage.
    const bool wants_address = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
    VkMemoryAllocateFlagsInfo flags_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };

    const VkMemoryAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = wants_address ? &flags_info : NULL,
        .allocationSize = requirements.size,
        .memoryTypeIndex = find_memory_type(device, requirements.memoryTypeBits, required,
                                            preferred),
    };
    VK_CHECK(vkAllocateMemory(device->device, &allocate_info, NULL, &buffer.memory));
    VK_CHECK(vkBindBufferMemory(device->device, buffer.handle, buffer.memory, 0));

    if (kind != GPU_MEMORY_DEVICE) {
        VK_CHECK(vkMapMemory(device->device, buffer.memory, 0, VK_WHOLE_SIZE, 0,
                             &buffer.mapped));
    }

    if (wants_address) {
        const VkBufferDeviceAddressInfo address_info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = buffer.handle,
        };
        buffer.address = vkGetBufferDeviceAddress(device->device, &address_info);
    }

    return buffer;
}

void gpu_buffer_destroy(gpu_device_t *device, gpu_buffer_t *buffer)
{
    if (buffer->mapped) {
        vkUnmapMemory(device->device, buffer->memory);
    }
    if (buffer->handle) {
        vkDestroyBuffer(device->device, buffer->handle, NULL);
    }
    if (buffer->memory) {
        vkFreeMemory(device->device, buffer->memory, NULL);
    }
    memset(buffer, 0, sizeof(*buffer));
}

gpu_image_t gpu_image_create(gpu_device_t *device, VkExtent2D extent, VkFormat format,
                             VkImageUsageFlags usage)
{
    gpu_image_t image = {.format = format, .extent = extent};

    const VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VK_CHECK(vkCreateImage(device->device, &image_info, NULL, &image.handle));

    VkMemoryRequirements requirements;
    vkGetImageMemoryRequirements(device->device, image.handle, &requirements);

    const VkMemoryAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size,
        .memoryTypeIndex = find_memory_type(device, requirements.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
    };
    VK_CHECK(vkAllocateMemory(device->device, &allocate_info, NULL, &image.memory));
    VK_CHECK(vkBindImageMemory(device->device, image.handle, image.memory, 0));

    const VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image.handle,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    VK_CHECK(vkCreateImageView(device->device, &view_info, NULL, &image.view));

    return image;
}

void gpu_image_destroy(gpu_device_t *device, gpu_image_t *image)
{
    if (image->view) {
        vkDestroyImageView(device->device, image->view, NULL);
    }
    if (image->handle) {
        vkDestroyImage(device->device, image->handle, NULL);
    }
    if (image->memory) {
        vkFreeMemory(device->device, image->memory, NULL);
    }
    memset(image, 0, sizeof(*image));
}

// ---------------------------------------------------------------------------
// immediate submission
// ---------------------------------------------------------------------------

void gpu_uploader_init(gpu_uploader_t *uploader, gpu_device_t *device)
{
    memset(uploader, 0, sizeof(*uploader));
    uploader->device = device;

    const VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = device->universal_family,
    };
    VK_CHECK(vkCreateCommandPool(device->device, &pool_info, NULL, &uploader->pool));

    const VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = uploader->pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VK_CHECK(vkAllocateCommandBuffers(device->device, &alloc_info, &uploader->cmd));

    const VkFenceCreateInfo fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(device->device, &fence_info, NULL, &uploader->fence));
}

void gpu_uploader_free(gpu_uploader_t *uploader)
{
    if (!uploader->device) {
        return;
    }
    VkDevice device = uploader->device->device;

    vkDestroyFence(device, uploader->fence, NULL);
    vkDestroyCommandPool(device, uploader->pool, NULL);
    memset(uploader, 0, sizeof(*uploader));
}

VkCommandBuffer gpu_upload_begin(gpu_uploader_t *uploader)
{
    VK_CHECK(vkResetCommandPool(uploader->device->device, uploader->pool, 0));

    const VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(uploader->cmd, &begin_info));
    return uploader->cmd;
}

void gpu_upload_end(gpu_uploader_t *uploader)
{
    VK_CHECK(vkEndCommandBuffer(uploader->cmd));

    const VkCommandBufferSubmitInfo cmd_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = uploader->cmd,
    };
    const VkSubmitInfo2 submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmd_info,
    };
    VK_CHECK(vkQueueSubmit2(uploader->device->universal_queue, 1, &submit, uploader->fence));

    VK_CHECK(vkWaitForFences(uploader->device->device, 1, &uploader->fence, VK_TRUE,
                             UINT64_MAX));
    VK_CHECK(vkResetFences(uploader->device->device, 1, &uploader->fence));
}

void gpu_buffer_upload(gpu_uploader_t *uploader, gpu_buffer_t *dst, const void *data,
                       VkDeviceSize size)
{
    if (size == 0) {
        return;
    }

    // Host-visible buffers can be written straight through the persistent mapping.
    if (dst->mapped) {
        memcpy(dst->mapped, data, (size_t)size);
        return;
    }

    gpu_buffer_t staging = gpu_buffer_create(uploader->device, size,
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                             GPU_MEMORY_UPLOAD);
    memcpy(staging.mapped, data, (size_t)size);

    VkCommandBuffer cmd = gpu_upload_begin(uploader);
    const VkBufferCopy region = {.size = size};
    vkCmdCopyBuffer(cmd, staging.handle, dst->handle, 1, &region);
    gpu_upload_end(uploader);

    gpu_buffer_destroy(uploader->device, &staging);
}

void gpu_image_upload(gpu_uploader_t *uploader, gpu_image_t *dst, const void *data,
                      VkDeviceSize size)
{
    if (size == 0) {
        return;
    }

    gpu_buffer_t staging = gpu_buffer_create(uploader->device, size,
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                             GPU_MEMORY_UPLOAD);
    memcpy(staging.mapped, data, (size_t)size);

    VkCommandBuffer cmd = gpu_upload_begin(uploader);

    // A fresh image is in UNDEFINED and holds nothing; the copy overwrites every texel.
    gpu_cmd_image_barrier(cmd, &(gpu_image_barrier_t){
        .image = dst->handle,
        .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
        .new_layout = VK_IMAGE_LAYOUT_GENERAL,
        .src_stage = VK_PIPELINE_STAGE_2_NONE,
        .src_access = VK_ACCESS_2_NONE,
        .dst_stage = VK_PIPELINE_STAGE_2_COPY_BIT,
        .dst_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
    });

    const VkBufferImageCopy2 region = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
        // Zeroed bufferRowLength/bufferImageHeight mean "tightly packed to imageExtent".
        .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
        .imageExtent = {dst->extent.width, dst->extent.height, 1},
    };
    const VkCopyBufferToImageInfo2 copy = {
        .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
        .srcBuffer = staging.handle,
        .dstImage = dst->handle,
        .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .regionCount = 1,
        .pRegions = &region,
    };
    vkCmdCopyBufferToImage2(cmd, &copy);

    // ALL_COMMANDS on the destination side because this layer does not know which stage the
    // caller will eventually sample from.
    gpu_cmd_image_barrier(cmd, &(gpu_image_barrier_t){
        .image = dst->handle,
        .old_layout = VK_IMAGE_LAYOUT_GENERAL,
        .new_layout = VK_IMAGE_LAYOUT_GENERAL,
        .src_stage = VK_PIPELINE_STAGE_2_COPY_BIT,
        .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dst_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dst_access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
    });

    gpu_upload_end(uploader);
    gpu_buffer_destroy(uploader->device, &staging);
}
