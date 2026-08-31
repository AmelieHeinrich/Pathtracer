#pragma once

#include "gpu.h"

// The response curve applied to scene-referred radiance on its way to the display.
//
// Order matters only in that TONEMAP_NONE is 0, so a zeroed struct means "do what the
// renderer did before this pass existed". The shader switches on these values, so the two
// sides have to agree -- see `Curve` in shaders/tonemap.slang.
typedef enum tonemap_curve_t {
    // Clip to [0,1]. Not a tonemapper at all: this is the naked clamp renderer_screenshot's
    // encode_srgb has always applied, kept so the spectral work can still be judged against
    // an image nothing has shaped.
    TONEMAP_NONE = 0,
    // Sobotka's AgX. The default because it desaturates on its way to white instead of
    // clipping channels independently, which is what a 5778 K sun disc against a blue sky
    // does to a naive curve.
    TONEMAP_AGX,
    // Narkowicz's fit to the ACES RRT+ODT. Contrastier than AgX and skews hue on bright
    // saturated colour, but it is the reference most images are judged against.
    TONEMAP_ACES,
    // x / (1 + x). No shoulder worth the name; useful mainly as a sanity check, because it is
    // the one curve simple enough to invert in your head.
    TONEMAP_REINHARD,
    TONEMAP_CURVE_COUNT,
} tonemap_curve_t;

// Names for the overlay's combo box, indexed by tonemap_curve_t.
extern const char *const TONEMAP_CURVE_NAMES[TONEMAP_CURVE_COUNT];

// Like denoise_settings_t, deliberately outside pt_settings_t: this shapes the averaged image
// on its way to the screen and cannot invalidate a single sample that went into it, so
// changing it must not restart the accumulation.
typedef struct tonemap_settings_t {
    bool enabled;
    tonemap_curve_t curve;
    // Stops. Applied as exp2(exposure) before the curve, so +1 is twice the light.
    float exposure;
} tonemap_settings_t;

typedef struct tonemap_t {
    gpu_device_t *device; // not owned

    // Written directly by the overlay, read once per tonemap_record.
    tonemap_settings_t settings;

    // Same format as the tracer's output image, so the screenshot readback decodes this or
    // that one with the same half float path.
    gpu_image_t target;
    // False until the first write to a freshly created image, which discards rather than
    // preserves. That is the only thing this decides.
    bool target_initialised;

    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkPipeline pipeline;
} tonemap_t;

void tonemap_init(tonemap_t *tonemap, gpu_device_t *device);
void tonemap_free(tonemap_t *tonemap);

// Rebuilds the destination image. Cheap no-op when the extent is unchanged.
void tonemap_resize(tonemap_t *tonemap, VkExtent2D extent);

// Recreates the pipeline from the .spv on disk, which the caller has just recompiled.
// Returns false and keeps the running pipeline on failure.
bool tonemap_reload_shaders(tonemap_t *tonemap);

// Maps `color` from scene-referred to display-referred and returns the image holding the
// result -- or `color` itself when the pass is switched off, so the caller's blit needs no
// special case. `color` must already be barriered for a compute shader read; the returned
// image is left ready for both a blit read and a transfer copy.
//
// The result stays *linear*: the swapchain is an sRGB format and the blit does the encode in
// hardware, so applying a transfer function here would apply it twice.
const gpu_image_t *tonemap_record(tonemap_t *tonemap, VkCommandBuffer cmd,
                                  const gpu_image_t *color);
