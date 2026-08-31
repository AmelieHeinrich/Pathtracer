#pragma once

#include "gpu.h"

// The a-trous filter's tunables, driven by the overlay.
//
// Deliberately *not* part of pt_settings_t: renderer_record memcmps that struct and restarts
// the accumulation on any change, which is right for anything the tracer does and exactly
// wrong for a post-process. Filtering the averaged image cannot invalidate the samples that
// went into it, so toggling this keeps every one of them.
typedef struct denoise_settings_t {
    bool enabled;
    // Passes over the image, tap spacing doubling each time. Four reaches a 33 pixel support.
    uint32_t iterations;
    // Luminance tolerance. Small preserves detail and noise alike; large flattens both.
    float phi_color;
    // Exponent on the cosine between normals. Larger is a harder crease.
    float phi_normal;
    // Depth tolerance in world units, scaled by the tap spacing inside the shader.
    float phi_depth;
} denoise_settings_t;

typedef struct denoise_t {
    gpu_device_t *device; // not owned

    // Written directly by the overlay, read once per denoise_record.
    denoise_settings_t settings;

    // Ping-pong pair. Iteration 0 reads the caller's image and writes pong[0]; every
    // iteration after that reads what the last one wrote. Same format as the tracer's output
    // image so the screenshot and PFM readbacks decode either one with the same code.
    gpu_image_t pong[2];
    // Cleared by a resize: the first write to a freshly created image discards rather than
    // preserves, and that is the only thing the flag decides.
    bool pong_initialised[2];

    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkPipeline pipeline;
} denoise_t;

void denoise_init(denoise_t *denoise, gpu_device_t *device);
void denoise_free(denoise_t *denoise);

// Rebuilds the ping-pong images. Cheap no-op when the extent is unchanged.
void denoise_resize(denoise_t *denoise, VkExtent2D extent);

// Recreates the pipeline from the .spv on disk, which the caller has just recompiled.
// Returns false and keeps the running pipeline on failure, like renderer_reload_shaders.
bool denoise_reload_shaders(denoise_t *denoise);

// Filters `color` and returns the image holding the result -- which is `color` itself when
// the filter is switched off, so the caller's blit needs no special case. `color` must
// already be barriered for a compute shader read; the returned image is left ready for both
// a blit read and a transfer copy.
const gpu_image_t *denoise_record(denoise_t *denoise, VkCommandBuffer cmd,
                                  const gpu_image_t *color, const gpu_image_t *normal,
                                  const gpu_image_t *depth);
