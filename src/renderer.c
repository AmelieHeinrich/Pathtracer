#include "renderer.h"

#include "gpu_internal.h"

#include <string.h>

#define PT_OUTPUT_FORMAT VK_FORMAT_R16G16B16A16_SFLOAT
// Full float: the accumulator keeps adding samples for as long as the camera holds still,
// and half would start dropping the small ones long before it converges.
#define PT_ACCUM_FORMAT VK_FORMAT_R32G32B32A32_SFLOAT
// A distance, not a clip space depth: the debug pass compares it against its own distance
// from the camera, which avoids ever having to invert a projection.
#define PT_DEPTH_FORMAT VK_FORMAT_R32_SFLOAT
// One entity index per pixel. Integer, so it must never be filtered or interpolated.
#define PT_PICK_FORMAT VK_FORMAT_R32_UINT
#define PT_SPIRV_PATH PT_SHADER_DIR "/pathtracer.spv"
#define PT_SLANG_PATH PT_SHADER_SRC_DIR "/pathtracer.slang"

// Starting point for pt_settings_t::max_bounces; the overlay takes it from here.
#define PT_DEFAULT_MAX_BOUNCES 5

// Index 0 is the ordinary miss shader the integrator's rays use; index 1 is the shadow ray's,
// which only has to record that nothing was in the way.
static const char *const PT_MISS_ENTRIES[] = {"miss", "miss_shadow"};

// One record per hit group, indexed by pt_hit_group_t: this order is what the instances'
// instanceShaderBindingTableRecordOffset values select between. The three procedural groups
// differ only in their intersection shader and share one closest hit shader.
static const gpu_hit_group_t PT_HIT_GROUPS[PT_HIT_GROUP_COUNT] = {
    [PT_HIT_GROUP_MESH] = {.closest_hit = "closesthit_mesh"},
    [PT_HIT_GROUP_SPHERE] = {.closest_hit = "closesthit_procedural",
                             .intersection = "isect_sphere"},
    [PT_HIT_GROUP_CYLINDER] = {.closest_hit = "closesthit_procedural",
                               .intersection = "isect_cylinder"},
    [PT_HIT_GROUP_CONE] = {.closest_hit = "closesthit_procedural",
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
    const VkDescriptorImageInfo depth_info = {
        .imageView = renderer->depth.view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo pick_info = {
        .imageView = renderer->pick.view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
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
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4},
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
    // Storage because raygen writes it, sampled because the debug pass reads it from a
    // fragment shader, where a writable storage image would need fragmentStoresAndAtomics.
    renderer->depth = gpu_image_create(renderer->device, extent, PT_DEPTH_FORMAT,
                                       VK_IMAGE_USAGE_STORAGE_BIT |
                                           VK_IMAGE_USAGE_SAMPLED_BIT);
    // TRANSFER_SRC so renderer_pick can copy a single pixel of it back to the host.
    renderer->pick = gpu_image_create(renderer->device, extent, PT_PICK_FORMAT,
                                      VK_IMAGE_USAGE_STORAGE_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
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
        .unlit = 0,
        .sky_intensity = 1.0f,
    };

    gpu_uploader_init(&renderer->uploader, device);
    create_descriptor_objects(renderer);

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
    gpu_image_destroy(gpu, &renderer->pick);
    gpu_image_destroy(gpu, &renderer->depth);
    gpu_image_destroy(gpu, &renderer->accum);
    gpu_image_destroy(gpu, &renderer->output);

    vkDestroyDescriptorPool(gpu->device, renderer->descriptor_pool, NULL);
    vkDestroyDescriptorSetLayout(gpu->device, renderer->set_layout, NULL);

    pt_scene_free(&renderer->scene, gpu);

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

    gpu_image_destroy(renderer->device, &renderer->pick);
    gpu_image_destroy(renderer->device, &renderer->depth);
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

void renderer_sync_scene(renderer_t *renderer)
{
    if (pt_scene_sync(&renderer->scene, renderer->device, &renderer->uploader)) {
        // Binding 0 names the acceleration structure by handle, and the rebuild made a new
        // one. The images are unchanged, but rewriting all three costs nothing.
        write_descriptors(renderer);
    }
}

bool renderer_reload_shaders(renderer_t *renderer)
{
    char command[1024];
    // Mirrors the build rule in xmake.lua, including -O0 (this slang build cannot load
    // its own spirv-opt) with the system optimiser applied afterwards when present.
    snprintf(command, sizeof(command),
             "\"%s\" \"%s\" -target spirv -profile spirv_1_5 -emit-spirv-directly "
             "-fvk-use-entrypoint-name -fvk-use-scalar-layout -O0 -o \"%s\" && "
             "{ command -v spirv-opt >/dev/null && "
             "spirv-opt --scalar-block-layout -O \"%s\" -o \"%s\" || true; }",
             PT_SLANGC, PT_SLANG_PATH, PT_SPIRV_PATH, PT_SPIRV_PATH, PT_SPIRV_PATH);

    if (system(command) != 0) {
        fprintf(stderr, "reload: slangc failed, keeping the running pipeline\n");
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
        .src_stage = VK_PIPELINE_STAGE_2_BLIT_BIT |
                     VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        .src_access = VK_ACCESS_2_NONE,
        .dst_stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        .dst_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    });

    // Unlike the output image the accumulator is read-modify-written, so this both orders
    // it after the previous frame's writes and keeps its contents. The one exception is a
    // restart, where the previous contents are about to be ignored and the image may not
    // even be in GENERAL yet because it was only just created.
    gpu_cmd_image_barrier(cmd, &(gpu_image_barrier_t){
        .image = renderer->accum.handle,
        .old_layout = renderer->accum_frames == 0 ? VK_IMAGE_LAYOUT_UNDEFINED
                                                  : VK_IMAGE_LAYOUT_GENERAL,
        .new_layout = VK_IMAGE_LAYOUT_GENERAL,
        .src_stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        .src_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dst_stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        .dst_access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    });

    // Both are fully overwritten every frame like `output`. They are consumed off the frame
    // path -- depth by the debug pass's fragment shader, pick by a host readback -- so the
    // source scope names those rather than the blit.
    const VkImage per_pixel_outputs[] = {renderer->depth.handle, renderer->pick.handle};
    for (uint32_t i = 0; i < PT_COUNT(per_pixel_outputs); ++i) {
        gpu_cmd_image_barrier(cmd, &(gpu_image_barrier_t){
            .image = per_pixel_outputs[i],
            .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            .new_layout = VK_IMAGE_LAYOUT_GENERAL,
            .src_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                         VK_PIPELINE_STAGE_2_COPY_BIT |
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
        .camera = *camera,
        .aspect = (float)extent.width / (float)extent.height,
        .frame_index = renderer->accum_frames,
        .light_count = renderer->scene.light_count,
        .settings = renderer->settings,
    };
    vkCmdPushConstants(cmd, renderer->pipeline.layout, GPU_RT_PUSH_CONSTANT_STAGES, 0,
                       sizeof(push), &push);
    ++renderer->accum_frames;

    const gpu_sbt_t *sbt = &renderer->pipeline.sbt;
    vkCmdTraceRaysKHR(cmd, &sbt->raygen, &sbt->miss, &sbt->hit, &sbt->callable, extent.width,
                      extent.height, 1);

    // Ray tracing writes -> blit reads.
    gpu_cmd_image_barrier(cmd, &(gpu_image_barrier_t){
        .image = renderer->output.handle,
        .old_layout = VK_IMAGE_LAYOUT_GENERAL,
        .new_layout = VK_IMAGE_LAYOUT_GENERAL,
        .src_stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        .src_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dst_stage = VK_PIPELINE_STAGE_2_BLIT_BIT,
        .dst_access = VK_ACCESS_2_TRANSFER_READ_BIT,
    });

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
        .srcImage = renderer->output.handle,
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
