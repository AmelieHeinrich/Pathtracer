#include "renderer.h"

#include "bluenoise.h"
#include "gpu_internal.h"
#include "spectral.h"

#include <stb/stb_image_write.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PT_OUTPUT_FORMAT VK_FORMAT_R16G16B16A16_SFLOAT
// Full float: the accumulator keeps adding samples for as long as the camera holds still,
// and half would start dropping the small ones long before it converges.
#define PT_ACCUM_FORMAT VK_FORMAT_R32G32B32A32_SFLOAT
// A distance, not a clip space depth: the debug pass compares it against its own distance
// from the camera, which avoids ever having to invert a projection.
#define PT_DEPTH_FORMAT VK_FORMAT_R32_SFLOAT
// Running sum of squared sample magnitude, which with the sum and the count in `accum` is
// everything needed to estimate a pixel's own variance. Full float for the same reason the
// accumulator is: it is a sum that grows without bound.
#define PT_MOMENT_FORMAT VK_FORMAT_R32_SFLOAT
// One entity index per pixel. Integer, so it must never be filtered or interpolated.
#define PT_PICK_FORMAT VK_FORMAT_R32_UINT
// Half float is ample for a unit vector; the denoiser only ever takes a dot product of two.
#define PT_NORMAL_FORMAT VK_FORMAT_R16G16B16A16_SFLOAT
#define PT_SPIRV_PATH PT_SHADER_DIR "/pathtracer.spv"
#define PT_SLANG_PATH PT_SHADER_SRC_DIR "/pathtracer.slang"
// The denoiser's module. Compiled here as well as by the build, so a hot reload rebuilds
// both shaders rather than leaving one of them stale.
#define PT_ATROUS_SPIRV_PATH PT_SHADER_DIR "/atrous.spv"
#define PT_ATROUS_SLANG_PATH PT_SHADER_SRC_DIR "/atrous.slang"
#define PT_TONEMAP_SPIRV_PATH PT_SHADER_DIR "/tonemap.spv"
#define PT_TONEMAP_SLANG_PATH PT_SHADER_SRC_DIR "/tonemap.slang"

// Starting point for pt_settings_t::max_bounces; the overlay takes it from here.
#define PT_DEFAULT_MAX_BOUNCES 5

// Index 0 is the ordinary miss shader the integrator's rays use; index 1 is the shadow ray's,
// which only has to record that nothing was in the way.
static const char *const PT_MISS_ENTRIES[] = {"miss", "miss_shadow"};

// One record per hit group, indexed by pt_hit_group_t: this order is what the instances'
// instanceShaderBindingTableRecordOffset values select between. The three procedural groups
// differ only in their intersection shader and share one closest hit shader.
//
// The any-hit shaders exist solely so a shadow ray can pass through glass; see the comment on
// anyhit_shadow_mesh. They cost nothing on primary rays, which never invoke them because the
// geometry is built opaque.
static const gpu_hit_group_t PT_HIT_GROUPS[PT_HIT_GROUP_COUNT] = {
    [PT_HIT_GROUP_MESH] = {.closest_hit = "closesthit_mesh",
                           .any_hit = "anyhit_shadow_mesh"},
    [PT_HIT_GROUP_SPHERE] = {.closest_hit = "closesthit_procedural",
                             .any_hit = "anyhit_shadow_procedural",
                             .intersection = "isect_sphere"},
    [PT_HIT_GROUP_CYLINDER] = {.closest_hit = "closesthit_procedural",
                               .any_hit = "anyhit_shadow_procedural",
                               .intersection = "isect_cylinder"},
    [PT_HIT_GROUP_CONE] = {.closest_hit = "closesthit_procedural",
                           .any_hit = "anyhit_shadow_procedural",
                           .intersection = "isect_cone"},
};

// ---------------------------------------------------------------------------
// pipeline and descriptors
// ---------------------------------------------------------------------------

static gpu_rt_pipeline_desc_t pipeline_desc(const renderer_t *renderer)
{
    return (gpu_rt_pipeline_desc_t){
        .spirv_path = PT_SPIRV_PATH,
        .raygen_entry = "raygen",
        .miss_entries = PT_MISS_ENTRIES,
        .miss_count = PT_COUNT(PT_MISS_ENTRIES),
        .hit_groups = PT_HIT_GROUPS,
        .hit_group_count = PT_COUNT(PT_HIT_GROUPS),
        .set_layout = renderer->set_layout,
        .push_constant_size = sizeof(pt_push_constants_t),
        // The integrator loops in raygen instead of tracing from inside a hit shader, so
        // one level is all the traversal ever needs.
        .max_recursion = 1,
    };
}

// Points the descriptor set at the current TLAS and images. Called on init and again
// whenever the images are recreated by a resize.
static void write_descriptors(renderer_t *renderer)
{
    const VkWriteDescriptorSetAccelerationStructureKHR accel_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &renderer->scene.tlas.handle,
    };
    const VkDescriptorImageInfo output_info = {
        .imageView = renderer->output.view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo accum_info = {
        .imageView = renderer->accum.view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo moment_info = {
        .imageView = renderer->moment.view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo depth_info = {
        .imageView = renderer->depth.view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo pick_info = {
        .imageView = renderer->pick.view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo normal_info = {
        .imageView = renderer->normal.view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    // Not one of the render targets: a fixed table uploaded once, so unlike the five above it
    // survives a resize untouched. It is rewritten here anyway because this function rewrites
    // the whole set, which costs nothing and keeps the two lists in step.
    const VkDescriptorImageInfo bluenoise_info = {
        .imageView = pt_bluenoise_image()->view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    // The one piece of bulk scene data that is a descriptor rather than a device address, and
    // only because there is no room left for a fourth address in the push constants. Like the
    // instance and light tables it is allocated once at full capacity, so this write stays
    // valid across every sync and every resize.
    const VkDescriptorBufferInfo emitter_info = {
        .buffer = renderer->scene.emitter_data.handle,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };

    const VkWriteDescriptorSet writes[] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = &accel_write,
            .dstSet = renderer->set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = renderer->set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &output_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = renderer->set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &accum_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = renderer->set,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &depth_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = renderer->set,
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &pick_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = renderer->set,
            .dstBinding = 5,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &normal_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = renderer->set,
            .dstBinding = 6,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &bluenoise_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = renderer->set,
            .dstBinding = 7,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &emitter_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = renderer->set,
            .dstBinding = 8,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &moment_info,
        },
    };
    vkUpdateDescriptorSets(renderer->device->device, PT_COUNT(writes), writes, 0, NULL);
}

static void create_descriptor_objects(renderer_t *renderer)
{
    // Deliberately minimal: everything bulky travels as a device address in the push
    // constant, so this layout stays stable as the scene grows. Raygen alone is enough of
    // a stage mask because the integrator loops there; the hit shaders only ever read the
    // instance table, which is a device address rather than a descriptor.
    const VkDescriptorSetLayoutBinding bindings[] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },
        {
            .binding = 4,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },
        {
            .binding = 5,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },
        {
            .binding = 6,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },
        {
            .binding = 7,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },
        {
            .binding = 8,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },
    };
    const VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = PT_COUNT(bindings),
        .pBindings = bindings,
    };
    VK_CHECK(vkCreateDescriptorSetLayout(renderer->device->device, &layout_info, NULL,
                                         &renderer->set_layout));

    const VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
    };
    const VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = PT_COUNT(pool_sizes),
        .pPoolSizes = pool_sizes,
    };
    VK_CHECK(vkCreateDescriptorPool(renderer->device->device, &pool_info, NULL,
                                    &renderer->descriptor_pool));

    const VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = renderer->descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &renderer->set_layout,
    };
    VK_CHECK(vkAllocateDescriptorSets(renderer->device->device, &alloc_info, &renderer->set));
}

static void create_targets(renderer_t *renderer, VkExtent2D extent)
{
    renderer->output = gpu_image_create(renderer->device, extent, PT_OUTPUT_FORMAT,
                                        VK_IMAGE_USAGE_STORAGE_BIT |
                                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    renderer->accum = gpu_image_create(renderer->device, extent, PT_ACCUM_FORMAT,
                                       VK_IMAGE_USAGE_STORAGE_BIT);
    // Read-modify-written alongside the accumulator and by nothing else, so it needs neither
    // sampling nor transfer.
    renderer->moment = gpu_image_create(renderer->device, extent, PT_MOMENT_FORMAT,
                                        VK_IMAGE_USAGE_STORAGE_BIT);
    // Storage because raygen writes it, sampled because the debug pass reads it from a
    // fragment shader, where a writable storage image would need fragmentStoresAndAtomics.
    renderer->depth = gpu_image_create(renderer->device, extent, PT_DEPTH_FORMAT,
                                       VK_IMAGE_USAGE_STORAGE_BIT |
                                           VK_IMAGE_USAGE_SAMPLED_BIT);
    // TRANSFER_SRC so renderer_pick can copy a single pixel of it back to the host.
    renderer->pick = gpu_image_create(renderer->device, extent, PT_PICK_FORMAT,
                                      VK_IMAGE_USAGE_STORAGE_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    // Storage only: raygen writes it and the denoiser's compute pass reads it, and neither
    // of those needs it sampled.
    renderer->normal = gpu_image_create(renderer->device, extent, PT_NORMAL_FORMAT,
                                        VK_IMAGE_USAGE_STORAGE_BIT);
    // Both post-process passes size their own images from the render target's.
    denoise_resize(&renderer->denoise, extent);
    tonemap_resize(&renderer->tonemap, extent);
    // Nothing has been blitted at this size yet, and the old pointers may name destroyed
    // images. renderer_record replaces both before anything reads them.
    renderer->display = &renderer->output;
    renderer->display_linear = &renderer->output;
    // A fresh accumulator holds nothing worth keeping.
    renderer->accum_frames = 0;
}

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

void renderer_init(renderer_t *renderer, gpu_device_t *device)
{
    memset(renderer, 0, sizeof(*renderer));
    renderer->device = device;
    renderer->settings = (pt_settings_t){
        .max_bounces = PT_DEFAULT_MAX_BOUNCES,
        .samples_per_frame = 1,
        // Both variance reducers on by default. Neither changes what the image converges to
        // -- the clamp's bound widens without limit and the adaptive budget only decides
        // where samples go -- so there is no reason to make the good behaviour opt-in.
        .flags = PT_FLAG_CLAMP | PT_FLAG_ADAPTIVE,
        .sky_intensity = 1.0f,
        .turbidity = 3.0f,
        .sun_azimuth = 200.0f,
        // Roughly where the old hardcoded sun lobe sat, so the default framing of the shipped
        // scenes is not thrown away by the switch to an authored sun.
        .sun_elevation = 44.0f,
        // The real sun. Affordable now only because sun_lighting samples the disc explicitly;
        // before that it had to be a 14 degree lobe to be found at all.
        .sun_angular_diameter = 0.53f,
    };

    gpu_uploader_init(&renderer->uploader, device);

    // Before the scene, because an entity's colour means nothing without the tables that turn
    // it into a spectrum. A failure here is reported and survivable -- pt_spectral_init leaves
    // an achromatic table behind rather than a null address -- so it is deliberately not fatal.
    pt_spectral_init(device, &renderer->uploader, PT_ASSET_DIR "/bin");

    // Before create_descriptor_objects, whose set names the atlas by image view. Survivable
    // in the same way the spectral tables are: it falls back to white noise and says so.
    pt_bluenoise_init(device, &renderer->uploader, PT_ASSET_DIR);

    create_descriptor_objects(renderer);

    // Before create_targets, which asks both to size their images.
    denoise_init(&renderer->denoise, device);
    tonemap_init(&renderer->tonemap, device);

    // The built-in showcase, which main.c then replaces with a scene file when one loads.
    // Starting from it means a missing or broken file leaves something on screen.
    pt_scene_default(&renderer->scene);
    pt_scene_init(&renderer->scene, device, &renderer->uploader);

    // Placeholder size; the first renderer_resize replaces it with the real extent.
    create_targets(renderer, (VkExtent2D){1, 1});
    write_descriptors(renderer);

    const gpu_rt_pipeline_desc_t desc = pipeline_desc(renderer);
    if (!gpu_rt_pipeline_create(device, &renderer->uploader, &desc, &renderer->pipeline)) {
        gpu_fatal("could not create the ray tracing pipeline from %s", PT_SPIRV_PATH);
    }
}

void renderer_free(renderer_t *renderer)
{
    if (!renderer->device) {
        return;
    }
    gpu_device_t *gpu = renderer->device;

    gpu_rt_pipeline_destroy(gpu, &renderer->pipeline);
    tonemap_free(&renderer->tonemap);
    denoise_free(&renderer->denoise);
    gpu_image_destroy(gpu, &renderer->normal);
    gpu_image_destroy(gpu, &renderer->pick);
    gpu_image_destroy(gpu, &renderer->depth);
    gpu_image_destroy(gpu, &renderer->moment);
    gpu_image_destroy(gpu, &renderer->accum);
    gpu_image_destroy(gpu, &renderer->output);

    vkDestroyDescriptorPool(gpu->device, renderer->descriptor_pool, NULL);
    vkDestroyDescriptorSetLayout(gpu->device, renderer->set_layout, NULL);

    pt_scene_free(&renderer->scene, gpu);
    pt_bluenoise_free(gpu);
    pt_spectral_free(gpu);

    gpu_uploader_free(&renderer->uploader);
    memset(renderer, 0, sizeof(*renderer));
}

void renderer_resize(renderer_t *renderer, VkExtent2D extent)
{
    if (renderer->output.extent.width == extent.width &&
        renderer->output.extent.height == extent.height) {
        return;
    }

    // The old images may still be referenced by frames in flight.
    VK_CHECK(vkDeviceWaitIdle(renderer->device->device));

    gpu_image_destroy(renderer->device, &renderer->normal);
    gpu_image_destroy(renderer->device, &renderer->pick);
    gpu_image_destroy(renderer->device, &renderer->depth);
    gpu_image_destroy(renderer->device, &renderer->moment);
    gpu_image_destroy(renderer->device, &renderer->accum);
    gpu_image_destroy(renderer->device, &renderer->output);
    create_targets(renderer, extent);
    write_descriptors(renderer);
}

void renderer_reset_accumulation(renderer_t *renderer)
{
    renderer->accum_frames = 0;
}

uint32_t renderer_pick(renderer_t *renderer, uint32_t x, uint32_t y)
{
    if (x >= renderer->pick.extent.width || y >= renderer->pick.extent.height) {
        return UINT32_MAX;
    }

    // The image holds what the last completed frame wrote, so that frame has to be done --
    // and there is no cheaper way to say so from outside a recording frame. A stall here is
    // acceptable because this only ever runs on a click.
    VK_CHECK(vkDeviceWaitIdle(renderer->device->device));

    gpu_buffer_t staging = gpu_buffer_create(renderer->device, sizeof(uint32_t),
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                             GPU_MEMORY_READBACK);

    VkCommandBuffer cmd = gpu_upload_begin(&renderer->uploader);
    const VkBufferImageCopy2 region = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
        .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
        .imageOffset = {(int32_t)x, (int32_t)y, 0},
        .imageExtent = {1, 1, 1},
    };
    const VkCopyImageToBufferInfo2 copy = {
        .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
        .srcImage = renderer->pick.handle,
        .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .dstBuffer = staging.handle,
        .regionCount = 1,
        .pRegions = &region,
    };
    vkCmdCopyImageToBuffer2(cmd, &copy);
    gpu_upload_end(&renderer->uploader); // submits and waits

    uint32_t value = 0;
    memcpy(&value, staging.mapped, sizeof(value));
    gpu_buffer_destroy(renderer->device, &staging);

    // Stored one-based so 0 could mean "the ray escaped".
    return value == 0 ? UINT32_MAX : value - 1;
}

// PT_OUTPUT_FORMAT is a half float format, and there is no half type in C11. Decoding by hand
// is a dozen lines and keeps the readback free of any dependency on compiler extensions.
static float half_to_float(uint16_t half)
{
    const uint32_t sign = (uint32_t)(half >> 15) << 31;
    const uint32_t exponent = (half >> 10) & 0x1Fu;
    const uint32_t mantissa = half & 0x3FFu;

    uint32_t bits;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            bits = sign; // +-0
        } else {
            // Subnormal as a half, but normal as a float: shift the mantissa up until its
            // leading bit falls off the top, and pay for each shift with the exponent.
            uint32_t shifted = mantissa;
            uint32_t shift = 0u;
            while ((shifted & 0x400u) == 0u) {
                shifted <<= 1;
                ++shift;
            }
            bits = sign | ((127u - 15u - shift + 1u) << 23) | ((shifted & 0x3FFu) << 13);
        }
    } else if (exponent == 31u) {
        bits = sign | 0x7F800000u | (mantissa << 13); // infinity or NaN
    } else {
        // The only difference is the exponent bias: 15 for half, 127 for float.
        bits = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }

    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

// The encode the sRGB swapchain format performs in hardware on the blit. Done by hand here
// because a PNG has to carry display-referred values, and this image is linear.
static uint8_t encode_srgb(float linear)
{
    // NaN fails every comparison, so testing for the *inside* of the range and defaulting to
    // zero keeps a stray NaN from reaching the cast below as garbage.
    float value = 0.0f;
    if (linear > 0.0f) {
        value = linear < 1.0f ? linear : 1.0f;
    }

    const float encoded =
        value <= 0.0031308f ? value * 12.92f : 1.055f * powf(value, 1.0f / 2.4f) - 0.055f;
    return (uint8_t)(encoded * 255.0f + 0.5f);
}

bool renderer_screenshot(renderer_t *renderer, const char *path)
{
    // Whatever the last frame put on screen, denoised or not. Its format matches the output
    // image's either way, so the half float decode below is the same for both.
    const gpu_image_t *displayed = renderer->display;
    const VkExtent2D extent = displayed->extent;
    if (extent.width == 0u || extent.height == 0u) {
        fprintf(stderr, "screenshot: nothing has been rendered yet\n");
        return false;
    }

    const VkDeviceSize pixels = (VkDeviceSize)extent.width * extent.height;
    const VkDeviceSize size = pixels * 8u; // RGBA, 16 bits a channel

    // Same reasoning as renderer_pick: the image holds what the last completed frame wrote,
    // and a stall is the only way to say so from outside a recording frame. This runs on a
    // keypress, so the cost is invisible.
    VK_CHECK(vkDeviceWaitIdle(renderer->device->device));

    gpu_buffer_t staging = gpu_buffer_create(renderer->device, size,
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                             GPU_MEMORY_READBACK);

    VkCommandBuffer cmd = gpu_upload_begin(&renderer->uploader);
    const VkBufferImageCopy2 region = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
        .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
        .imageExtent = {extent.width, extent.height, 1},
    };
    const VkCopyImageToBufferInfo2 copy = {
        .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
        .srcImage = displayed->handle,
        // The render target never leaves GENERAL -- see the barriers in renderer_record.
        .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .dstBuffer = staging.handle,
        .regionCount = 1,
        .pRegions = &region,
    };
    vkCmdCopyImageToBuffer2(cmd, &copy);
    gpu_upload_end(&renderer->uploader); // submits and waits

    // Three bytes a pixel: the alpha channel raygen writes is a constant 1 and carries
    // nothing worth storing.
    uint8_t *rgb = gpu_alloc((size_t)pixels * 3u);

    const uint16_t *source = staging.mapped;
    for (VkDeviceSize i = 0; i < pixels; ++i) {
        for (uint32_t channel = 0; channel < 3u; ++channel) {
            rgb[i * 3u + channel] = encode_srgb(half_to_float(source[i * 4u + channel]));
        }
    }

    const int written = stbi_write_png(path, (int)extent.width, (int)extent.height, 3, rgb,
                                       (int)extent.width * 3);
    free(rgb);
    gpu_buffer_destroy(renderer->device, &staging);

    if (!written) {
        fprintf(stderr, "screenshot: could not write '%s'\n", path);
        return false;
    }

    printf("screenshot: wrote %s (%ux%u, %u samples)\n", path, extent.width, extent.height,
           renderer->accum_frames * renderer->settings.samples_per_frame);
    return true;
}

// The same readback as above, but written as a PFM: linear, full float, no sRGB encode and no
// quantisation. Comparing two builds means measuring differences of a few percent against
// Monte Carlo noise, and eight bits through a gamma curve cannot resolve that -- so the eye
// gets the PNG and the diff script gets this.
bool renderer_capture_pfm(renderer_t *renderer, const char *path)
{
    // Stops before the tonemapper, unlike the screenshot: a PFM has to stay scene-referred
    // to be worth diffing. See the comment on renderer_capture_pfm.
    const gpu_image_t *displayed = renderer->display_linear;
    const VkExtent2D extent = displayed->extent;
    if (extent.width == 0u || extent.height == 0u) {
        fprintf(stderr, "capture: nothing has been rendered yet\n");
        return false;
    }

    const VkDeviceSize pixels = (VkDeviceSize)extent.width * extent.height;

    VK_CHECK(vkDeviceWaitIdle(renderer->device->device));

    gpu_buffer_t staging = gpu_buffer_create(renderer->device, pixels * 8u,
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                             GPU_MEMORY_READBACK);

    VkCommandBuffer cmd = gpu_upload_begin(&renderer->uploader);
    const VkBufferImageCopy2 region = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
        .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
        .imageExtent = {extent.width, extent.height, 1},
    };
    const VkCopyImageToBufferInfo2 copy = {
        .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
        .srcImage = displayed->handle,
        .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .dstBuffer = staging.handle,
        .regionCount = 1,
        .pRegions = &region,
    };
    vkCmdCopyImageToBuffer2(cmd, &copy);
    gpu_upload_end(&renderer->uploader);

    FILE *file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "capture: could not open '%s'\n", path);
        gpu_buffer_destroy(renderer->device, &staging);
        return false;
    }

    // A negative scale is PFM's way of saying the samples are little endian.
    fprintf(file, "PF\n%u %u\n-1.0\n", extent.width, extent.height);

    const uint16_t *source = staging.mapped;
    float *row = gpu_alloc((size_t)extent.width * 3u * sizeof(float));
    bool ok = true;

    // PFM rows run bottom to top, which is the opposite of the image's own order.
    for (uint32_t y = 0; y < extent.height && ok; ++y) {
        const uint16_t *line = source + (VkDeviceSize)(extent.height - 1u - y) * extent.width * 4u;
        for (uint32_t x = 0; x < extent.width; ++x) {
            for (uint32_t channel = 0; channel < 3u; ++channel) {
                row[x * 3u + channel] = half_to_float(line[x * 4u + channel]);
            }
        }
        ok = fwrite(row, sizeof(float), (size_t)extent.width * 3u, file) ==
             (size_t)extent.width * 3u;
    }

    free(row);
    ok = fclose(file) == 0 && ok;
    gpu_buffer_destroy(renderer->device, &staging);

    if (!ok) {
        fprintf(stderr, "capture: could not write '%s'\n", path);
        return false;
    }

    printf("screenshot: wrote %s (%ux%u, %u samples)\n", path, extent.width, extent.height,
           renderer->accum_frames * renderer->settings.samples_per_frame);
    return true;
}

void renderer_sync_scene(renderer_t *renderer)
{
    if (pt_scene_sync(&renderer->scene, renderer->device, &renderer->uploader)) {
        // Binding 0 names the acceleration structure by handle, and the rebuild made a new
        // one. The images are unchanged, but rewriting all three costs nothing.
        write_descriptors(renderer);
    }
}

// One .slang module to one .spv. Mirrors the build rule in xmake.lua, including -O0 (this
// slang build cannot load its own spirv-opt) with the system optimiser applied afterwards
// when present.
static bool compile_slang(const char *source, const char *spirv)
{
    char command[1024];
    snprintf(command, sizeof(command),
             "\"%s\" \"%s\" -target spirv -profile spirv_1_5 -emit-spirv-directly "
             "-fvk-use-entrypoint-name -fvk-use-scalar-layout -O0 -o \"%s\" && "
             "{ command -v spirv-opt >/dev/null && "
             "spirv-opt --scalar-block-layout -O \"%s\" -o \"%s\" || true; }",
             PT_SLANGC, source, spirv, spirv, spirv);

    if (system(command) != 0) {
        fprintf(stderr, "reload: slangc failed on %s\n", source);
        return false;
    }
    return true;
}

bool renderer_reload_shaders(renderer_t *renderer)
{
    // Both modules are compiled before either pipeline is touched, so a broken denoise
    // shader cannot leave the tracer half reloaded.
    if (!compile_slang(PT_SLANG_PATH, PT_SPIRV_PATH) ||
        !compile_slang(PT_ATROUS_SLANG_PATH, PT_ATROUS_SPIRV_PATH) ||
        !compile_slang(PT_TONEMAP_SLANG_PATH, PT_TONEMAP_SPIRV_PATH)) {
        fprintf(stderr, "reload: keeping the running pipelines\n");
        return false;
    }

    // Build the replacement fully before disturbing anything that is rendering.
    const gpu_rt_pipeline_desc_t desc = pipeline_desc(renderer);
    gpu_rt_pipeline_t rebuilt = {0};
    if (!gpu_rt_pipeline_create(renderer->device, &renderer->uploader, &desc, &rebuilt)) {
        fprintf(stderr, "reload: pipeline creation failed, keeping the running pipeline\n");
        return false;
    }

    VK_CHECK(vkDeviceWaitIdle(renderer->device->device));
    gpu_rt_pipeline_destroy(renderer->device, &renderer->pipeline);
    renderer->pipeline = rebuilt;

    // The denoiser is a post-process, so a failure here leaves a perfectly good image on
    // screen; report it and carry on rather than failing the whole reload.
    if (!denoise_reload_shaders(&renderer->denoise)) {
        fprintf(stderr, "reload: the tracer reloaded but the denoiser did not\n");
    }
    if (!tonemap_reload_shaders(&renderer->tonemap)) {
        fprintf(stderr, "reload: the tracer reloaded but the tonemapper did not\n");
    }

    // Whatever has accumulated so far came out of the old shaders.
    renderer_reset_accumulation(renderer);

    printf("reload: shaders reloaded\n");
    fflush(stdout);
    return true;
}

// ---------------------------------------------------------------------------
// recording
// ---------------------------------------------------------------------------

void renderer_record(renderer_t *renderer, gpu_frame_t *frame, const pt_camera_t *camera)
{
    VkCommandBuffer cmd = frame->cmd;
    const VkExtent2D extent = renderer->output.extent;

    // Any camera movement, or any change the overlay made to the settings, invalidates every
    // sample taken so far. The camera is built by pt_camera_look_at from the same inputs each
    // frame and pt_settings_t has no padding, so a bytewise compare is exact for both.
    if (memcmp(camera, &renderer->last_camera, sizeof(*camera)) != 0 ||
        memcmp(&renderer->settings, &renderer->last_settings,
               sizeof(renderer->settings)) != 0 ||
        renderer->scene.synced_revision != renderer->last_scene_revision) {
        renderer_reset_accumulation(renderer);
        renderer->last_camera = *camera;
        renderer->last_settings = renderer->settings;
        renderer->last_scene_revision = renderer->scene.synced_revision;
    }

    // One output image is shared by every frame in flight, and the timeline only waits
    // for frame N-2, so frame N's writes can overlap frame N-1's blit read of the same
    // image. Naming BLIT as the source stage is what orders this write after that read;
    // a write-after-read hazard needs execution ordering only, hence no source access.
    // UNDEFINED discards the previous contents, which every pixel overwrites anyway.
    gpu_cmd_image_barrier(cmd, &(gpu_image_barrier_t){
        .image = renderer->output.handle,
        .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
        .new_layout = VK_IMAGE_LAYOUT_GENERAL,
        // COMPUTE as well, because with the denoiser on the previous frame read this image
        // from its first iteration rather than blitting it.
        .src_stage = VK_PIPELINE_STAGE_2_BLIT_BIT |
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                     VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        .src_access = VK_ACCESS_2_NONE,
        .dst_stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        .dst_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    });

    // Unlike the output image the accumulator is read-modify-written, so this both orders
    // it after the previous frame's writes and keeps its contents. The one exception is a
    // restart, where the previous contents are about to be ignored and the image may not
    // even be in GENERAL yet because it was only just created.
    // The moment image is the accumulator's second half and shares its lifetime exactly, so it
    // takes the same barrier -- including the restart case, where neither has meaningful
    // contents and the layout may still be UNDEFINED.
    const VkImage accumulators[] = {renderer->accum.handle, renderer->moment.handle};
    for (uint32_t i = 0; i < PT_COUNT(accumulators); ++i) {
        gpu_cmd_image_barrier(cmd, &(gpu_image_barrier_t){
            .image = accumulators[i],
            .old_layout = renderer->accum_frames == 0 ? VK_IMAGE_LAYOUT_UNDEFINED
                                                      : VK_IMAGE_LAYOUT_GENERAL,
            .new_layout = VK_IMAGE_LAYOUT_GENERAL,
            .src_stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .src_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .dst_stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .dst_access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        });
    }

    // These three are consumed off the frame path -- depth by the debug pass's fragment shader
    // and by the denoiser, pick by a host readback, normal by the denoiser -- so the source
    // scope names those rather than the blit.
    //
    // Their contents are *kept*, unlike `output`. With an adaptive sample budget a pixel that
    // has already settled traces nothing, and so writes none of these; it has to keep the
    // geometry it wrote earlier. That is safe because any camera or scene change restarts the
    // accumulation, and the restart frame traces every pixel -- which is also the one frame
    // where there is nothing worth keeping, hence the same test the accumulators use.
    const VkImage per_pixel_outputs[] = {renderer->depth.handle, renderer->pick.handle,
                                         renderer->normal.handle};
    for (uint32_t i = 0; i < PT_COUNT(per_pixel_outputs); ++i) {
        gpu_cmd_image_barrier(cmd, &(gpu_image_barrier_t){
            .image = per_pixel_outputs[i],
            .old_layout = renderer->accum_frames == 0 ? VK_IMAGE_LAYOUT_UNDEFINED
                                                      : VK_IMAGE_LAYOUT_GENERAL,
            .new_layout = VK_IMAGE_LAYOUT_GENERAL,
            .src_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                         VK_PIPELINE_STAGE_2_COPY_BIT |
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                         VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .src_access = VK_ACCESS_2_NONE,
            .dst_stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .dst_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        });
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                      renderer->pipeline.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            renderer->pipeline.layout, 0, 1, &renderer->set, 0, NULL);

    const pt_push_constants_t push = {
        .instances = renderer->scene.instance_data.address,
        .lights = renderer->scene.light_data.address,
        .spectra = pt_spectral_address(),
        .camera = *camera,
        .frame_index = renderer->accum_frames,
        .light_count = renderer->scene.light_count,
        .emitter_count = renderer->scene.emitter_count,
        .settings = renderer->settings,
    };
    vkCmdPushConstants(cmd, renderer->pipeline.layout, GPU_RT_PUSH_CONSTANT_STAGES, 0,
                       sizeof(push), &push);
    ++renderer->accum_frames;

    const gpu_sbt_t *sbt = &renderer->pipeline.sbt;
    vkCmdTraceRaysKHR(cmd, &sbt->raygen, &sbt->miss, &sbt->hit, &sbt->callable, extent.width,
                      extent.height, 1);

    const bool denoising = renderer->denoise.settings.enabled;
    // Whether *anything* consumes the traced image in a compute shader before the blit. Both
    // post-process passes are optional and either one can be first, so this asks about the
    // chain as a whole rather than about the denoiser alone.
    const bool post_processing = denoising || renderer->tonemap.settings.enabled;

    // Ray tracing writes -> whoever reads the image next. That is a compute pass when one is
    // switched on, and the blit below when neither is. The two G-buffers only matter to the
    // denoiser, so they are only barriered when it will actually read them.
    const VkImage traced[] = {renderer->output.handle, renderer->normal.handle,
                              renderer->depth.handle};
    const uint32_t traced_count = denoising ? PT_COUNT(traced) : 1u;
    for (uint32_t i = 0; i < traced_count; ++i) {
        gpu_cmd_image_barrier(cmd, &(gpu_image_barrier_t){
            .image = traced[i],
            .old_layout = VK_IMAGE_LAYOUT_GENERAL,
            .new_layout = VK_IMAGE_LAYOUT_GENERAL,
            .src_stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .src_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .dst_stage = post_processing ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                         : VK_PIPELINE_STAGE_2_BLIT_BIT,
            .dst_access = post_processing ? VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                          : VK_ACCESS_2_TRANSFER_READ_BIT,
        });
    }

    // The post-process chain. Each pass returns its input untouched when switched off, so
    // every combination of the two toggles falls out without a special case -- and the blit
    // below simply takes whatever the last one returned.
    //
    // Recorded here rather than after the blit because the blit is what puts it on screen.
    renderer->display = denoise_record(&renderer->denoise, cmd, &renderer->output,
                                       &renderer->normal, &renderer->depth);
    // Captured before the tonemapper: a PFM has to stay scene-referred.
    renderer->display_linear = renderer->display;
    renderer->display = tonemap_record(&renderer->tonemap, cmd, renderer->display);

    // The swapchain image arrives in GENERAL from gpu_frame_begin, and is sRGB, so the
    // blit converts this linear float image into sRGB for free.
    const VkImageBlit2 region = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
        .srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
        .srcOffsets = {{0, 0, 0}, {(int32_t)extent.width, (int32_t)extent.height, 1}},
        .dstSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
        .dstOffsets = {{0, 0, 0},
                       {(int32_t)frame->extent.width, (int32_t)frame->extent.height, 1}},
    };
    const VkBlitImageInfo2 blit = {
        .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
        .srcImage = renderer->display->handle,
        .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .dstImage = frame->image,
        .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .regionCount = 1,
        .pRegions = &region,
        .filter = VK_FILTER_NEAREST,
    };
    vkCmdBlitImage2(cmd, &blit);

    // No barrier needed after this: gpu_frame_end's transition to PRESENT_SRC_KHR takes
    // ALL_COMMANDS/MEMORY_WRITE as its source scope, which already covers the blit.
}
