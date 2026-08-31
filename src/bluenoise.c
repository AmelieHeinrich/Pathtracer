#include "bluenoise.h"

#include "gpu_internal.h"

#include <stb/stb_image.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Christoph Peters' free blue noise textures, whose LDR_RGBA variant carries four independent
// channels in one 8 bit image. Two realisations are loaded so the sampler has eight channels
// to hand out; they must be different files, or dimensions 0-3 and 4-7 would correlate.
static const char *const PT_BLUENOISE_FILES[PT_BLUENOISE_TILES] = {
    "LDR_RGBA_0.png",
    "LDR_RGBA_1.png",
};

// A module singleton for the same reason src/spectral.c is one: this is a fixed table the
// whole renderer shares, not something a scene owns.
static gpu_image_t g_image;
static bool g_loaded = false;

// The white noise the atlas falls back to. Deliberately the same generator the shader uses,
// so the fallback is exactly the sampling the renderer had before blue noise existed.
static uint32_t pcg_hash(uint32_t value)
{
    const uint32_t state = value * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

static void fill_white_noise(uint8_t *pixels)
{
    const uint32_t count = PT_BLUENOISE_WIDTH * PT_BLUENOISE_HEIGHT * 4u;
    for (uint32_t i = 0; i < count; ++i) {
        pixels[i] = (uint8_t)(pcg_hash(i) >> 24);
    }
}

// Reads one tile into `pixels` at the horizontal offset for `tile`. Returns false and leaves
// the region untouched on any failure, having already explained itself.
static bool load_tile(const char *dir, uint32_t tile, uint8_t *pixels)
{
    char path[512];
    // The upstream set names its directories with an underscore -- assets/128_128 -- not an x.
    snprintf(path, sizeof(path), "%s/%u_%u/%s", dir, PT_BLUENOISE_TILE, PT_BLUENOISE_TILE,
             PT_BLUENOISE_FILES[tile]);

    int width = 0;
    int height = 0;
    int channels = 0;
    // Forced to 4 channels: the file is RGBA already, but asking makes the stride below true
    // by construction rather than by inspection.
    stbi_uc *data = stbi_load(path, &width, &height, &channels, 4);
    if (!data) {
        // The overwhelmingly likely cause is a Git LFS checkout that never had `git lfs pull`
        // run, which leaves a ~130 byte text stub in place of every image. Worth naming,
        // because "unsupported file format" is a baffling thing to read about a .png.
        fprintf(stderr, "bluenoise: could not read %s (%s)\n", path, stbi_failure_reason());
        fprintf(stderr, "bluenoise: if these are Git LFS pointer stubs, run `git lfs pull`\n");
        return false;
    }

    if (width != (int)PT_BLUENOISE_TILE || height != (int)PT_BLUENOISE_TILE) {
        fprintf(stderr, "bluenoise: %s is %dx%d, expected %ux%u\n", path, width, height,
                PT_BLUENOISE_TILE, PT_BLUENOISE_TILE);
        stbi_image_free(data);
        return false;
    }

    // Copied row by row: the source is a tight 128 wide image and the destination is one half
    // of a 256 wide atlas, so the strides differ.
    for (uint32_t y = 0; y < PT_BLUENOISE_TILE; ++y) {
        uint8_t *dst = pixels + ((size_t)y * PT_BLUENOISE_WIDTH + tile * PT_BLUENOISE_TILE) * 4u;
        memcpy(dst, data + (size_t)y * PT_BLUENOISE_TILE * 4u, PT_BLUENOISE_TILE * 4u);
    }

    stbi_image_free(data);
    return true;
}

bool pt_bluenoise_init(gpu_device_t *device, gpu_uploader_t *uploader, const char *dir)
{
    if (g_loaded) {
        return true;
    }

    const VkDeviceSize size = (VkDeviceSize)PT_BLUENOISE_WIDTH * PT_BLUENOISE_HEIGHT * 4u;
    uint8_t *pixels = gpu_alloc((size_t)size);

    bool ok = true;
    for (uint32_t tile = 0; tile < PT_BLUENOISE_TILES; ++tile) {
        ok = load_tile(dir, tile, pixels) && ok;
    }

    // All or nothing: half a loaded atlas would correlate the dimensions that came from the
    // failed tile with each other in a way that is much worse than plain white noise.
    if (!ok) {
        fprintf(stderr, "bluenoise: falling back to white noise\n");
        fill_white_noise(pixels);
    }

    // UNORM, so the shader's texel fetch already yields [0,1] with no conversion. Sampled
    // usage is not asked for: nothing filters this, and a storage image needs no sampler.
    g_image = gpu_image_create(device, (VkExtent2D){PT_BLUENOISE_WIDTH, PT_BLUENOISE_HEIGHT},
                               VK_FORMAT_R8G8B8A8_UNORM,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    gpu_image_upload(uploader, &g_image, pixels, size);

    free(pixels);
    g_loaded = true;
    return ok;
}

void pt_bluenoise_free(gpu_device_t *device)
{
    if (!g_loaded) {
        return;
    }
    gpu_image_destroy(device, &g_image);
    g_loaded = false;
}

const gpu_image_t *pt_bluenoise_image(void)
{
    return &g_image;
}
