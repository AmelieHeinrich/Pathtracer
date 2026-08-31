#include "tonemap.h"

#include "gpu_internal.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define TONEMAP_SPIRV_PATH PT_SHADER_DIR "/tonemap.spv"

// Must match the tracer's output image and the denoiser's ping-pong pair: this pass sits at
// the end of a chain whose images the screenshot readback decodes with one half float path.
#define TONEMAP_FORMAT VK_FORMAT_R16G16B16A16_SFLOAT

// The shader's [numthreads]. Both sides have to agree, so the dispatch takes it from here.
#define TONEMAP_GROUP_SIZE 8

const char *const TONEMAP_CURVE_NAMES[TONEMAP_CURVE_COUNT] = {
    [TONEMAP_NONE] = "None",
    [TONEMAP_AGX] = "AgX",
    [TONEMAP_ACES] = "ACES",
    [TONEMAP_REINHARD] = "Reinhard",
};

// Matches `struct PushConstants` in shaders/tonemap.slang.
typedef struct tonemap_push_constants_t {
    uint32_t width;
    uint32_t height;
    uint32_t curve;
    float exposure_scale;
} tonemap_push_constants_t;

// ---------------------------------------------------------------------------
// pipeline
// ---------------------------------------------------------------------------

// Returns false rather than aborting, so a hot reload can survive a broken shader.
static bool create_pipeline(tonemap_t *tonemap, VkPipeline *out)
{
    VkShaderModule module = gpu_shader_module_load(tonemap->device, TONEMAP_SPIRV_PATH);
    if (!module) {
        return false;
    }

    const VkComputePipelineCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = module,
                // Named rather than "main": everything here is compiled with
                // -fvk-use-entrypoint-name.
                .pName = "tonemap",
            },
        .layout = tonemap->layout,
    };
    const VkResult result = vkCreateComputePipelines(tonemap->device->device, VK_NULL_HANDLE, 1,
                                                     &info, NULL, out);
    vkDestroyShaderModule(tonemap->device->device, module, NULL);

    if (result != VK_SUCCESS) {
        fprintf(stderr, "tonemap: vkCreateComputePipelines failed: %s\n",
                vk_result_string(result));
        return false;
    }
    return true;
}

static void create_layouts(tonemap_t *tonemap)
{
    // Pushed inline rather than allocated from a pool: the source is whatever the previous
    // pass returned, which changes with the denoise toggle, so there is nothing stable to
    // write a pooled set with. Same reasoning as the denoiser and the debug line pass.
    const VkDescriptorSetLayoutBinding bindings[] = {
        {
            .binding = 0, // scene-referred colour, read
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1, // display-referred colour, written
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo set_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
        .bindingCount = PT_COUNT(bindings),
        .pBindings = bindings,
    };
    VK_CHECK(vkCreateDescriptorSetLayout(tonemap->device->device, &set_layout_info, NULL,
                                         &tonemap->set_layout));

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .size = sizeof(tonemap_push_constants_t),
    };
    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &tonemap->set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    VK_CHECK(vkCreatePipelineLayout(tonemap->device->device, &layout_info, NULL,
                                    &tonemap->layout));
}

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

void tonemap_init(tonemap_t *tonemap, gpu_device_t *device)
{
    memset(tonemap, 0, sizeof(*tonemap));
    tonemap->device = device;

    tonemap->settings = (tonemap_settings_t){
        // On by default, unlike the denoiser: an unshaped image is the exception now, not the
        // rule, and TONEMAP_NONE is there for when it is wanted.
        .enabled = true,
        .curve = TONEMAP_AGX,
        .exposure = 0.0f,
    };

    create_layouts(tonemap);
    if (!create_pipeline(tonemap, &tonemap->pipeline)) {
        gpu_fatal("could not create the tonemap pipeline from %s", TONEMAP_SPIRV_PATH);
    }
}

void tonemap_free(tonemap_t *tonemap)
{
    if (!tonemap->device) {
        return;
    }
    VkDevice device = tonemap->device->device;

    gpu_image_destroy(tonemap->device, &tonemap->target);
    vkDestroyPipeline(device, tonemap->pipeline, NULL);
    vkDestroyPipelineLayout(device, tonemap->layout, NULL);
    vkDestroyDescriptorSetLayout(device, tonemap->set_layout, NULL);

    memset(tonemap, 0, sizeof(*tonemap));
}

void tonemap_resize(tonemap_t *tonemap, VkExtent2D extent)
{
    if (tonemap->target.extent.width == extent.width &&
        tonemap->target.extent.height == extent.height) {
        return;
    }

    // The caller has already waited for the device, but this must hold either way round.
    VK_CHECK(vkDeviceWaitIdle(tonemap->device->device));

    gpu_image_destroy(tonemap->device, &tonemap->target);
    // TRANSFER_SRC because a screenshot reads back whichever image was last put on screen,
    // and with this pass on that is this one.
    tonemap->target = gpu_image_create(tonemap->device, extent, TONEMAP_FORMAT,
                                       VK_IMAGE_USAGE_STORAGE_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    tonemap->target_initialised = false;
}

bool tonemap_reload_shaders(tonemap_t *tonemap)
{
    // Built in full before anything that is rendering is disturbed.
    VkPipeline rebuilt = VK_NULL_HANDLE;
    if (!create_pipeline(tonemap, &rebuilt)) {
        fprintf(stderr, "tonemap: keeping the running pipeline\n");
        return false;
    }

    VK_CHECK(vkDeviceWaitIdle(tonemap->device->device));
    vkDestroyPipeline(tonemap->device->device, tonemap->pipeline, NULL);
    tonemap->pipeline = rebuilt;
    return true;
}

// ---------------------------------------------------------------------------
// recording
// ---------------------------------------------------------------------------

const gpu_image_t *tonemap_record(tonemap_t *tonemap, VkCommandBuffer cmd,
                                  const gpu_image_t *color)
{
    const tonemap_settings_t settings = tonemap->settings;
    const VkExtent2D extent = color->extent;

    // Nothing recorded at all when the pass is off, so the caller's blit and readbacks see
    // exactly the image they saw before this pass existed.
    if (!settings.enabled || extent.width == 0u || extent.height == 0u ||
        tonemap->target.handle == VK_NULL_HANDLE) {
        return color;
    }

    // Discarding on the first use of a freshly created image, preserving afterwards -- though
    // "preserving" is academic, since every pixel is overwritten. What this barrier is really
    // for is ordering against the previous frame's blit and any screenshot copy that read it.
    gpu_cmd_image_barrier(cmd, &(gpu_image_barrier_t){
        .image = tonemap->target.handle,
        .old_layout = tonemap->target_initialised ? VK_IMAGE_LAYOUT_GENERAL
                                                  : VK_IMAGE_LAYOUT_UNDEFINED,
        .new_layout = VK_IMAGE_LAYOUT_GENERAL,
        .src_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                     VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
        .src_access = VK_ACCESS_2_NONE,
        .dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dst_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    });
    tonemap->target_initialised = true;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tonemap->pipeline);

    const VkDescriptorImageInfo src_info = {
        .imageView = color->view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo dst_info = {
        .imageView = tonemap->target.view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkWriteDescriptorSet writes[] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &src_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &dst_info,
        },
    };
    vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tonemap->layout, 0,
                           PT_COUNT(writes), writes);

    const tonemap_push_constants_t push = {
        .width = extent.width,
        .height = extent.height,
        .curve = (uint32_t)settings.curve,
        // Turned into a plain multiplier here rather than in the shader, where it would be
        // recomputed once per pixel for a value that is constant across the dispatch.
        .exposure_scale = exp2f(settings.exposure),
    };
    vkCmdPushConstants(cmd, tonemap->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                       &push);

    vkCmdDispatch(cmd, (extent.width + TONEMAP_GROUP_SIZE - 1u) / TONEMAP_GROUP_SIZE,
                  (extent.height + TONEMAP_GROUP_SIZE - 1u) / TONEMAP_GROUP_SIZE, 1);

    // Compute writes -> the caller's blit, and the copy a screenshot may make of the same
    // image later. Both are transfer reads, so one barrier covers them.
    gpu_cmd_image_barrier(cmd, &(gpu_image_barrier_t){
        .image = tonemap->target.handle,
        .old_layout = VK_IMAGE_LAYOUT_GENERAL,
        .new_layout = VK_IMAGE_LAYOUT_GENERAL,
        .src_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .src_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dst_stage = VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
        .dst_access = VK_ACCESS_2_TRANSFER_READ_BIT,
    });

    return &tonemap->target;
}
