#pragma once

#include "gpu.h"

// The blue noise tiles the sampler draws its low-dimensional prefix from.
//
// Two 128x128 RGBA tiles packed side by side into one 256x128 image, giving eight independent
// channels -- four per tile -- of which the integrator currently uses five. A 2D storage image
// rather than a buffer at a device address for three reasons: the push constant block has no
// room for another address, src/gpu.h exposes no sampler API, and an integer texel fetch is
// all blue noise ever wants anyway.
#define PT_BLUENOISE_TILE 128u
#define PT_BLUENOISE_TILES 2u
#define PT_BLUENOISE_WIDTH (PT_BLUENOISE_TILE * PT_BLUENOISE_TILES)
#define PT_BLUENOISE_HEIGHT PT_BLUENOISE_TILE

// Loads the tiles from `dir` and uploads them. Idempotent, and never fatal: on any failure --
// a missing file, a decode error, the wrong dimensions, or a Git LFS pointer stub where the
// image should be -- it fills the atlas with white noise instead and says so once. The shader
// path is identical either way, so a missing asset costs image quality and nothing else.
//
// Returns false when it fell back, for the caller to report; the atlas is usable regardless.
bool pt_bluenoise_init(gpu_device_t *device, gpu_uploader_t *uploader, const char *dir);
void pt_bluenoise_free(gpu_device_t *device);

// The atlas, for the descriptor write. Valid from pt_bluenoise_init until pt_bluenoise_free.
const gpu_image_t *pt_bluenoise_image(void);
