#pragma once

#include "gpu.h"
#include "pt_math.h" // pt_camera_t
#include "scene.h"

// The tunables the overlay drives. Every member is 4 bytes on purpose: renderer_record
// detects a change with a bytewise compare, which padding would make unreliable.
typedef struct pt_settings_t {
    uint32_t max_bounces;
    // Samples taken per pixel per frame. Accumulation across frames converges on its own, so
    // this only trades frame rate for how fast a still image settles.
    uint32_t samples_per_frame;
    uint32_t unlit; // non-zero: show albedo straight out of the first hit, no lighting
    // Scales the sky the miss shader returns. 0 turns it off entirely, leaving only the
    // explicit lights -- which is what you want to judge a lighting setup on its own.
    float sky_intensity;
} pt_settings_t;

// Matches `struct PushConstants` in shaders/pathtracer.slang. Bulk data reaches the shader
// as device addresses, so adding geometry never touches the descriptor set.
typedef struct pt_push_constants_t {
    VkDeviceAddress instances;
    VkDeviceAddress lights;
    pt_camera_t camera;
    float aspect;
    uint32_t frame_index;
    uint32_t light_count;
    pt_settings_t settings;
} pt_push_constants_t;

typedef struct renderer_t {
    gpu_device_t *device; // not owned
    gpu_uploader_t uploader;

    pt_scene_t scene;

    // Ray tracing target. Linear float, because the sRGB swapchain cannot take STORAGE
    // usage; the blit into it performs the encode.
    gpu_image_t output;
    // Running sum of radiance across every frame since the last reset. Full float, because
    // it has to keep adding samples without losing the small ones.
    gpu_image_t accum;
    // Distance from the camera to the primary hit, written once per frame regardless of the
    // sample count. Only the debug lines read it, to hide themselves behind scene geometry.
    gpu_image_t depth;
    // The entity under each pixel, plus one. Read back a pixel at a time by renderer_pick.
    gpu_image_t pick;
    uint32_t accum_frames;
    // What the last recorded frame ran with. Anything that differs invalidates every sample
    // taken so far.
    pt_camera_t last_camera;
    pt_settings_t last_settings;
    // Compared against the scene's *synced* revision, not its authored one: an edit must not
    // restart the accumulation until the frame that actually traces against the new TLAS.
    uint32_t last_scene_revision;

    // Written directly by the overlay; read once per renderer_record.
    pt_settings_t settings;

    VkDescriptorSetLayout set_layout;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet set;

    gpu_rt_pipeline_t pipeline;
} renderer_t;

void renderer_init(renderer_t *renderer, gpu_device_t *device);
void renderer_free(renderer_t *renderer);

// Rebuilds the output images and rewrites their descriptors. Cheap no-op when unchanged.
void renderer_resize(renderer_t *renderer, VkExtent2D extent);

// Recompiles the shaders and swaps the pipeline. Returns false and keeps the running
// pipeline when compilation or pipeline creation fails.
bool renderer_reload_shaders(renderer_t *renderer);

// Restarts the accumulation on the next recorded frame.
void renderer_reset_accumulation(renderer_t *renderer);

// Rebuilds the scene's GPU objects if it has been edited, and repoints the descriptor set at
// the new acceleration structure. Submits and waits internally, so it must be called outside
// a recording frame -- not between gpu_frame_begin and gpu_frame_end.
void renderer_sync_scene(renderer_t *renderer);

// The entity index under a framebuffer pixel, or UINT32_MAX for none. Reads back what the
// last completed frame traced, so it agrees exactly with what is on screen -- the same
// intersection shaders decided both. Stalls the device, so click only; and like
// renderer_sync_scene it must not be called between gpu_frame_begin and gpu_frame_end.
uint32_t renderer_pick(renderer_t *renderer, uint32_t x, uint32_t y);

// Any change to `camera` or to `renderer->settings` restarts the accumulation by itself.
void renderer_record(renderer_t *renderer, gpu_frame_t *frame, const pt_camera_t *camera);
