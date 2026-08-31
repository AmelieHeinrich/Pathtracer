#include "denoise.h"

#include "gpu_internal.h"

#include <stdio.h>
#include <string.h>

#define DENOISE_SPIRV_PATH PT_SHADER_DIR "/atrous.spv"

// Must match the tracer's output image: the filter reads that image on its first iteration
// and the screenshot and PFM readbacks decode whichever image ended up on screen with one
// half-float path, so the two cannot differ.
#define DENOISE_FORMAT VK_FORMAT_R16G16B16A16_SFLOAT

// The shader's [numthreads]. Both sides have to agree, so the dispatch takes it from here.
#define DENOISE_GROUP_SIZE 8

// The widest tap spacing the shader is given, which is 1 << (DENOISE_MAX_ITERATIONS - 1).
// Also the range the overlay's Iterations property is clamped to.
#define DENOISE_MAX_ITERATIONS 5

// Matches `struct PushConstants` in shaders/atrous.slang. Its own block rather than an
// extension of pt_push_constants_t, which is already within a few bytes of the 128 Vulkan
// guarantees.
typedef struct denoise_push_constants_t {
    uint32_t width;
    uint32_t height;
    uint32_t step;
    float phi_color;
    float phi_normal;
    float phi_depth;
} denoise_push_constants_t;

// ---------------------------------------------------------------------------
// pipeline
// ---------------------------------------------------------------------------

// Returns false rather than aborting, so a hot reload can survive a broken shader. The
// caller decides whether that is fatal: it is at init, and it is not on reload.
static bool create_pipeline(denoise_t *denoise, VkPipeline *out)
{
    VkShaderModule module = gpu_shader_module_load(denoise->device, DENOISE_SPIRV_PATH);
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
                // The module holds one entry point, but it is still named rather than "main":
                // everything here is compiled with -fvk-use-entrypoint-name.
                .pName = "atrous",
            },
        .layout = denoise->layout,
    };
    const VkResult result = vkCreateComputePipelines(denoise->device->device, VK_NULL_HANDLE, 1,
                                                     &info, NULL, out);
    vkDestroyShaderModule(denoise->device->device, module, NULL);

    if (result != VK_SUCCESS) {
        fprintf(stderr, "denoise: vkCreateComputePipelines failed: %s\n",
                vk_result_string(result));
        return false;
    }
    return true;
}

static void create_layouts(denoise_t *denoise)
{
    // Pushed inline rather than allocated from a pool: the source and destination swap every
    // iteration, so a pool would need one set per iteration per frame in flight to describe
    // what four inline writes describe for free. Same reasoning as the debug line pass.
    const VkDescriptorSetLayoutBinding bindings[] = {
        {
            .binding = 0, // source colour, read
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1, // destination colour, written
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2, // primary hit normal, read
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 3, // primary hit depth, read
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
    VK_CHECK(vkCreateDescriptorSetLayout(denoise->device->device, &set_layout_info, NULL,
                                         &denoise->set_layout));

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .size = sizeof(denoise_push_constants_t),
    };
    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &denoise->set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    VK_CHECK(vkCreatePipelineLayout(denoise->device->device, &layout_info, NULL,
                                    &denoise->layout));
}

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

void denoise_init(denoise_t *denoise, gpu_device_t *device)
{
    memset(denoise, 0, sizeof(*denoise));
    denoise->device = device;

    denoise->settings = (denoise_settings_t){
        // Off by default: the unfiltered image is the one that is actually converging, and
        // the filter is something you reach for when you want a frame to look at.
        .enabled = false,
        .iterations = 4,
        .phi_color = 0.5f,
        .phi_normal = 64.0f,
        .phi_depth = 0.05f,
    };

    create_layouts(denoise);
    if (!create_pipeline(denoise, &denoise->pipeline)) {
        gpu_fatal("could not create the denoise pipeline from %s", DENOISE_SPIRV_PATH);
    }
}

void denoise_free(denoise_t *denoise)
{
    if (!denoise->device) {
        return;
    }
    VkDevice device = denoise->device->device;

    for (uint32_t i = 0; i < PT_COUNT(denoise->pong); ++i) {
        gpu_image_destroy(denoise->device, &denoise->pong[i]);
    }
    vkDestroyPipeline(device, denoise->pipeline, NULL);
    vkDestroyPipelineLayout(device, denoise->layout, NULL);
    vkDestroyDescriptorSetLayout(device, denoise->set_layout, NULL);

    memset(denoise, 0, sizeof(*denoise));
}

void denoise_resize(denoise_t *denoise, VkExtent2D extent)
{
    if (denoise->pong[0].extent.width == extent.width &&
        denoise->pong[0].extent.height == extent.height) {
        return;
    }

    // The caller (renderer_resize) has already waited for the device, but this must hold
    // whichever way round the two are called.
    VK_CHECK(vkDeviceWaitIdle(denoise->device->device));

    for (uint32_t i = 0; i < PT_COUNT(denoise->pong); ++i) {
        gpu_image_destroy(denoise->device, &denoise->pong[i]);
        // TRANSFER_SRC because a screenshot or a PFM capture reads back whichever image was
        // last put on screen, and with the filter on that is one of these.
        denoise->pong[i] = gpu_image_create(denoise->device, extent, DENOISE_FORMAT,
                                            VK_IMAGE_USAGE_STORAGE_BIT |
                                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        denoise->pong_initialised[i] = false;
    }
}

bool denoise_reload_shaders(denoise_t *denoise)
{
    // Built in full before anything that is rendering is disturbed.
    VkPipeline rebuilt = VK_NULL_HANDLE;
    if (!create_pipeline(denoise, &rebuilt)) {
        fprintf(stderr, "denoise: keeping the running pipeline\n");
        return false;
    }

    VK_CHECK(vkDeviceWaitIdle(denoise->device->device));
    vkDestroyPipeline(denoise->device->device, denoise->pipeline, NULL);
    denoise->pipeline = rebuilt;
    return true;
}

// ---------------------------------------------------------------------------
// recording
// ---------------------------------------------------------------------------

const gpu_image_t *denoise_record(denoise_t *denoise, VkCommandBuffer cmd,
                                  const gpu_image_t *color, const gpu_image_t *normal,
                                  const gpu_image_t *depth)
{
    const denoise_settings_t settings = denoise->settings;
    const VkExtent2D extent = color->extent;

    // Nothing recorded at all when the filter is off, so the caller's blit and readbacks see
    // exactly the image they saw before this pass existed.
    if (!settings.enabled || settings.iterations == 0u || extent.width == 0u ||
        extent.height == 0u || denoise->pong[0].handle == VK_NULL_HANDLE) {
        return color;
    }

    const uint32_t iterations = settings.iterations > DENOISE_MAX_ITERATIONS
                                    ? DENOISE_MAX_ITERATIONS
                                    : settings.iterations;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, denoise->pipeline);

    const VkDescriptorImageInfo normal_info = {
        .imageView = normal->view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo depth_info = {
        .imageView = depth->view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };

    const gpu_image_t *src = color;
    gpu_image_t *dst = NULL;

    for (uint32_t i = 0; i < iterations; ++i) {
        const uint32_t slot = i & 1u;
        dst = &denoise->pong[slot];

        // Discarding on the first use of a freshly created image, preserving afterwards --
        // though "preserving" is academic, since every pixel is overwritten. What this
        // barrier is really for is ordering: against the previous frame's blit and any
        // screenshot copy that read this image, and against iteration i-1 having read it.
        gpu_cmd_image_barrier(cmd, &(gpu_image_barrier_t){
            .image = dst->handle,
            .old_layout = denoise->pong_initialised[slot] ? VK_IMAGE_LAYOUT_GENERAL
                                                          : VK_IMAGE_LAYOUT_UNDEFINED,
            .new_layout = VK_IMAGE_LAYOUT_GENERAL,
            .src_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                         VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
            .src_access = VK_ACCESS_2_NONE,
            .dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dst_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        });
        denoise->pong_initialised[slot] = true;

        // Iteration 0's source is the caller's image, which the caller has already barriered.
        if (i > 0u) {
            gpu_cmd_image_barrier(cmd, &(gpu_image_barrier_t){
                .image = src->handle,
                .old_layout = VK_IMAGE_LAYOUT_GENERAL,
                .new_layout = VK_IMAGE_LAYOUT_GENERAL,
                .src_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .src_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dst_access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            });
        }

        const VkDescriptorImageInfo src_info = {
            .imageView = src->view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkDescriptorImageInfo dst_info = {
            .imageView = dst->view,
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
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &normal_info,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstBinding = 3,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &depth_info,
            },
        };
        vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, denoise->layout, 0,
                               PT_COUNT(writes), writes);

        const denoise_push_constants_t push = {
            .width = extent.width,
            .height = extent.height,
            .step = 1u << i,
            // Halved per iteration, per the paper: the colour weight has to tighten as the
            // taps spread, or the widest passes smear detail the earlier ones preserved.
            .phi_color = settings.phi_color / (float)(1u << i),
            .phi_normal = settings.phi_normal,
            .phi_depth = settings.phi_depth,
        };
        vkCmdPushConstants(cmd, denoise->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                           &push);

        vkCmdDispatch(cmd, (extent.width + DENOISE_GROUP_SIZE - 1u) / DENOISE_GROUP_SIZE,
                      (extent.height + DENOISE_GROUP_SIZE - 1u) / DENOISE_GROUP_SIZE, 1);

        src = dst;
    }

    // Compute writes -> everything that may read this next: the tonemap pass chained after
    // it, the caller's blit, and the copy a screenshot or a PFM capture makes of it later.
    // Naming all three rather than only the transfer reads is what lets another compute pass
    // be appended to the chain without a hazard.
    gpu_cmd_image_barrier(cmd, &(gpu_image_barrier_t){
        .image = dst->handle,
        .old_layout = VK_IMAGE_LAYOUT_GENERAL,
        .new_layout = VK_IMAGE_LAYOUT_GENERAL,
        .src_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .src_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                     VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
        .dst_access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT,
    });

    return dst;
}
