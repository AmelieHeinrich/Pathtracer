#pragma once

#include "denoise.h"
#include "gpu.h"
#include "pt_math.h" // pt_camera_t
#include "scene.h"
#include "tonemap.h"

// Show albedo straight out of the first hit, with no light transport at all.
#define PT_FLAG_UNLIT (1u << 0)
// Progressive firefly clamping. Bounds how far above a pixel's running mean a single sample
// is allowed to land, with the bound widening as the sample count grows -- so the spikes that
// dominate an early frame are held back while the estimator stays consistent. See
// clamp_threshold in shaders/pathtracer.slang.
#define PT_FLAG_CLAMP (1u << 1)
// Spend samples where the variance is, rather than evenly. Purely temporal: it changes how
// many samples a pixel gets, never which pixels it is averaged with.
#define PT_FLAG_ADAPTIVE (1u << 2)

// The tunables the overlay drives. Every member is 4 bytes on purpose: renderer_record
// detects a change with a bytewise compare, which padding would make unreliable.
typedef struct pt_settings_t {
    uint32_t max_bounces;
    // Samples taken per pixel per frame. Accumulation across frames converges on its own, so
    // this only trades frame rate for how fast a still image settles. With PT_FLAG_ADAPTIVE
    // on this is the budget a *converging* pixel gets; a settled one takes none and a noisy
    // one takes more.
    uint32_t samples_per_frame;
    // The PT_FLAG_* bits above. One field rather than three, because push constants are
    // already at the 128 byte guarantee -- see PT_PUSH_CONSTANT_LIMIT below.
    uint32_t flags;
    // Scales the sky the miss shader returns. 0 turns it off entirely, leaving only the
    // explicit lights -- which is what you want to judge a lighting setup on its own.
    float sky_intensity;
    // The Preetham sky's own parameters. Turbidity is haze: 2 is a clear alpine day, 10 is
    // thick city air. The sun's position is authored here rather than in the scene file for
    // the same reason sky_intensity is -- it is a property of how the scene is being *looked
    // at*, not of what is in it.
    float turbidity;
    float sun_azimuth;   // degrees, clockwise from -Z
    float sun_elevation; // degrees above the horizon
    // The sun's angular diameter in degrees. 0.53 is the real sun; larger softens its shadow
    // edges. Because sun_radiance divides by the disc's solid angle this changes softness
    // without changing brightness, so it reads as one control rather than two.
    float sun_angular_diameter;
} pt_settings_t;

// The no-padding rule above is not a style note: renderer_record memcmps this struct to decide
// whether to restart the accumulation, and padding bytes are uninitialised. Adding a member
// that is not 4 bytes would make that comparison read uninitialised memory and restart the
// accumulation at random.
_Static_assert(sizeof(pt_settings_t) == 8 * sizeof(uint32_t),
               "pt_settings_t must stay padding-free; renderer_record compares it bytewise");

// Matches `struct PushConstants` in shaders/pathtracer.slang. Bulk data reaches the shader
// as device addresses, so adding geometry never touches the descriptor set.
typedef struct pt_push_constants_t {
    VkDeviceAddress instances;
    VkDeviceAddress lights;
    // The baked spectral tables. Kept adjacent to the other two addresses so its 8 byte
    // alignment does not open a hole in the middle of the block.
    VkDeviceAddress spectra;
    pt_camera_t camera;
    // No `aspect` here: raygen derives it from DispatchRaysDimensions, which is the same
    // extent this would have been computed from. The four bytes matter -- see the ceiling
    // below, which this block now sits exactly on.
    uint32_t frame_index;
    uint32_t light_count;
    // How many entries of the emitter table are live. The table itself is a descriptor rather
    // than a fourth device address precisely because this block has no room for one: it sits
    // exactly on the 128 byte guarantee as it is.
    uint32_t emitter_count;
    pt_settings_t settings;
} pt_push_constants_t;

// Vulkan only guarantees 128 bytes of push constants, and gpu_rt_pipeline_create builds its
// range from sizeof(pt_push_constants_t) without consulting the device. The spectral port
// spends most of what is left here, so the ceiling is worth stating where the struct is
// declared -- and worth failing the build over rather than discovering as corruption.
#define PT_PUSH_CONSTANT_LIMIT 128
_Static_assert(sizeof(pt_push_constants_t) <= PT_PUSH_CONSTANT_LIMIT,
               "push constants exceed the guaranteed minimum; move bulk data behind a "
               "device address instead");

typedef struct renderer_t {
    gpu_device_t *device; // not owned
    gpu_uploader_t uploader;

    pt_scene_t scene;

    // Ray tracing target. Linear float, because the sRGB swapchain cannot take STORAGE
    // usage; the blit into it performs the encode.
    gpu_image_t output;
    // Running sum of radiance across every frame since the last reset, with the number of
    // samples that went into it in alpha. Full float, because it has to keep adding samples
    // without losing the small ones -- and because with adaptive sampling the count is no
    // longer the same for every pixel, so it has to be stored rather than derived.
    gpu_image_t accum;
    // Running sum of squared sample magnitude, the second moment that turns `accum` into a
    // per-pixel variance estimate. Read and written by raygen alone.
    gpu_image_t moment;
    // Distance from the camera to the primary hit, written once per frame regardless of the
    // sample count. Only the debug lines read it, to hide themselves behind scene geometry.
    gpu_image_t depth;
    // The entity under each pixel, plus one. Read back a pixel at a time by renderer_pick.
    gpu_image_t pick;
    // World space normal at the primary hit, oriented against the ray. Written for the
    // denoiser, which needs it to tell a crease from a flat surface at the same depth.
    gpu_image_t normal;
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

    // The a-trous post-process. Its own settings live inside it rather than in
    // pt_settings_t, so switching it on does not restart the accumulation.
    denoise_t denoise;
    // Scene-referred radiance to display-referred, chained after the denoiser. Its settings
    // live inside it for the same reason the denoiser's do.
    tonemap_t tonemap;
    // The image the last recorded frame actually blitted: the last link of the
    // output -> denoise -> tonemap chain that was switched on. renderer_screenshot reads
    // this, so a PNG always matches what was on screen.
    const gpu_image_t *display;
    // The same chain, stopped before the tonemapper. renderer_capture_pfm reads this instead:
    // a PFM exists to be diffed against another build, which means it has to stay
    // scene-referred, linear and unclamped. A display-referred PFM would silently invalidate
    // every comparison tools/compare_images.py is used for.
    const gpu_image_t *display_linear;

    VkDescriptorSetLayout set_layout;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet set;

    gpu_rt_pipeline_t pipeline;
} renderer_t;

void renderer_init(renderer_t *renderer, gpu_device_t *device);
void renderer_free(renderer_t *renderer);

// Rebuilds the output images and rewrites their descriptors. Cheap no-op when unchanged.
void renderer_resize(renderer_t *renderer, VkExtent2D extent);

// Recompiles the shaders and swaps the pipelines -- both the tracer's and the denoiser's.
// Returns false and keeps the running pipelines when compilation or pipeline creation fails.
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

// Writes the converged image to `path` as a PNG, applying the same linear-to-sRGB encode the
// swapchain blit gets from its format -- so the file matches what is on screen, overlay
// aside. Reads back the render target rather than the swapchain precisely so the UI and the
// gizmos stay out of it, and follows the denoise toggle for the same reason: what is captured
// is what was displayed.
//
// Stalls the device and shares renderer_pick's constraint: outside a recording frame only.
// Returns false and explains itself on failure; losing a screenshot must never take the
// renderer down with it.
bool renderer_screenshot(renderer_t *renderer, const char *path);

// The render as a PFM: linear, full float, unencoded and unquantised. This is the one to diff
// two builds with -- the differences worth measuring are a few percent against Monte Carlo
// noise, which does not survive eight bits through a gamma curve.
//
// Unlike the screenshot this stops before the tonemapper, so it stays scene-referred whatever
// the overlay is set to. It does follow the denoiser, which is a filter over the same
// quantity rather than a change of what the quantity means.
bool renderer_capture_pfm(renderer_t *renderer, const char *path);

// Any change to `camera` or to `renderer->settings` restarts the accumulation by itself.
void renderer_record(renderer_t *renderer, gpu_frame_t *frame, const pt_camera_t *camera);
