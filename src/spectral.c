#include "spectral.h"

#include "gpu_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Mirrors the header tools/bake_spectral.py writes, byte for byte. The trailing reserved words
// are what pad it to the 128 bytes the baker zero-fills to.
typedef struct pt_spectral_header_t {
    char magic[8]; // "PTSPEC01", not NUL terminated
    uint32_t version;
    uint32_t resolution;
    uint32_t coefficient_count;
    uint32_t lambda_count;
    float lambda_min;
    float lambda_max;
    float daylight_luminance[3]; // Y0/Y1/Y2; see the sky in shaders/spectral.slangh
    uint32_t flags;
    char name[48];
    uint32_t reserved[8];
} pt_spectral_header_t;

_Static_assert(sizeof(pt_spectral_header_t) == 128,
               "the spectral header must match the one tools/bake_spectral.py writes");

#define PT_SPECTRAL_MAGIC "PTSPEC01"
#define PT_SPECTRAL_VERSION 2u

// The block layout documented in spectral.h, in floats. The leading 3 are the daylight
// luminances, which come from the header rather than the payload; everything from the 9 entry
// matrix onwards is the payload verbatim.
#define PT_SPECTRAL_PREFIX_FLOATS (3 + 9 + 2 + 1)
#define PT_SPECTRAL_CURVE_FLOATS (PT_SPECTRAL_LAMBDA_COUNT * 3)
#define PT_SPECTRAL_COEFF_FLOATS \
    (3 * PT_SPECTRAL_RESOLUTION * PT_SPECTRAL_RESOLUTION * PT_SPECTRAL_RESOLUTION * 3)
#define PT_SPECTRAL_TOTAL_FLOATS                                          \
    (PT_SPECTRAL_PREFIX_FLOATS + PT_SPECTRAL_RESOLUTION +                 \
     2 * PT_SPECTRAL_CURVE_FLOATS + PT_SPECTRAL_COEFF_FLOATS)

#define PT_SPECTRAL_CMF_OFFSET (PT_SPECTRAL_PREFIX_FLOATS + PT_SPECTRAL_RESOLUTION)

static gpu_buffer_t g_buffer;
static bool g_loaded;
// Kept host side purely for pt_spectral_blackbody_norm, which has to integrate against the
// same observer the shader uses. 5.6 KiB; not worth a round trip to the GPU to avoid.
static float g_cmf[PT_SPECTRAL_CURVE_FLOATS];

// ---------------------------------------------------------------------------
// fallback
// ---------------------------------------------------------------------------

// A neutral table, used when the blob will not load. Every colour comes back as its own
// luminance in grey, which keeps the renderer running and the image legible while making it
// obvious at a glance that something is wrong. The alternative -- a null device address --
// faults the GPU, and this codebase's rule is that a bad asset never stops the renderer.
static void fill_achromatic(float *floats)
{
    // The daylight luminances and z nodes still have to be plausible: the sky reads the first
    // three, and the lookup searches the nodes.
    floats[0] = 1.0f;
    floats[1] = 0.0f;
    floats[2] = 0.0f;

    // Identity, so the resolve at least passes XYZ through instead of scrambling it.
    static const float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    memcpy(&floats[3], identity, sizeof(identity));
    floats[12] = 0.0f; // D65's M1/M2; irrelevant with a zeroed basis, but must be finite
    floats[13] = 0.0f;
    floats[14] = 1.0f; // the sun normalisation; 1 leaves the lobe at its authored brightness

    for (uint32_t i = 0; i < PT_SPECTRAL_RESOLUTION; ++i) {
        const float t = (float)i / (float)(PT_SPECTRAL_RESOLUTION - 1u);
        floats[PT_SPECTRAL_PREFIX_FLOATS + i] = t * t * (3.0f - 2.0f * t);
    }

    // Without the real colour matching functions there is nothing sensible to resolve
    // through, so leave the curves at zero and let the coefficients carry a flat grey. A
    // fallback render is for seeing that the renderer still runs, not for judging colour.
    memset(&floats[PT_SPECTRAL_CMF_OFFSET], 0,
           2 * PT_SPECTRAL_CURVE_FLOATS * sizeof(float));

    float *coefficients = &floats[PT_SPECTRAL_CMF_OFFSET + 2 * PT_SPECTRAL_CURVE_FLOATS];
    for (uint32_t i = 0; i < PT_SPECTRAL_COEFF_FLOATS / 3u; ++i) {
        // The reflectance a grid cell stands for is its z node; the sigmoid's inverse turns
        // that into a constant term with no slope and no curvature.
        const uint32_t level = (i / (PT_SPECTRAL_RESOLUTION * PT_SPECTRAL_RESOLUTION)) %
                               PT_SPECTRAL_RESOLUTION;
        const float reflectance =
            fminf(fmaxf(floats[PT_SPECTRAL_PREFIX_FLOATS + level], 1e-3f), 1.0f - 1e-3f);
        coefficients[i * 3 + 0] = 0.0f;
        coefficients[i * 3 + 1] = 0.0f;
        coefficients[i * 3 + 2] =
            (2.0f * reflectance - 1.0f) / (2.0f * sqrtf(reflectance * (1.0f - reflectance)));
    }
}

// ---------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------

// Reads the blob into `floats`, laid out as spectral.h documents. Every failure explains
// itself and returns false; the caller then ships the achromatic table instead.
static bool read_blob(const char *path, float *floats)
{
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "spectral: no table at '%s' -- run tools/bake_spectral.py\n", path);
        return false;
    }

    bool ok = false;
    pt_spectral_header_t header;
    do {
        if (fread(&header, sizeof(header), 1, file) != 1) {
            fprintf(stderr, "spectral: '%s' is too short to hold a header\n", path);
            break;
        }
        if (memcmp(header.magic, PT_SPECTRAL_MAGIC, sizeof(header.magic)) != 0) {
            fprintf(stderr, "spectral: '%s' is not a spectral table\n", path);
            break;
        }
        if (header.version != PT_SPECTRAL_VERSION) {
            fprintf(stderr,
                    "spectral: '%s' is version %u but this build reads %u -- re-run "
                    "tools/bake_spectral.py --force\n",
                    path, header.version, PT_SPECTRAL_VERSION);
            break;
        }
        // The shader computes every block offset from these as compile time constants, so a
        // blob that disagrees would be read at the wrong offsets rather than rejected.
        if (header.resolution != PT_SPECTRAL_RESOLUTION ||
            header.lambda_count != PT_SPECTRAL_LAMBDA_COUNT ||
            header.coefficient_count != 3u) {
            fprintf(stderr,
                    "spectral: '%s' is %ux%u but this build reads %ux%u -- re-run "
                    "tools/bake_spectral.py --force\n",
                    path, header.resolution, header.lambda_count, PT_SPECTRAL_RESOLUTION,
                    PT_SPECTRAL_LAMBDA_COUNT);
            break;
        }

        // Check the size before reading, so a wild header cannot ask for a wild read.
        // The blob carries everything except the three header luminances and the sun scale,
        // which are copied and computed in below.
        const long payload = (long)(PT_SPECTRAL_TOTAL_FLOATS - 4) * (long)sizeof(float);
        if (fseek(file, 0, SEEK_END) != 0 || ftell(file) != (long)sizeof(header) + payload) {
            fprintf(stderr, "spectral: '%s' is not the size its header describes\n", path);
            break;
        }
        if (fseek(file, (long)sizeof(header), SEEK_SET) != 0) {
            break;
        }

        // The luminance constants come from the header and lead the buffer; everything after
        // is the file's payload in its own order.
        memcpy(floats, header.daylight_luminance, sizeof(header.daylight_luminance));
        // The matrix and D65 weights sit between the header luminances and the sun scale, so
        // the payload is read in two runs around the slot this module fills itself.
        ok = fread(&floats[3], 11 * sizeof(float), 1, file) == 1 &&
             fread(&floats[15], (size_t)payload - 11 * (long)sizeof(float), 1, file) == 1;
        if (!ok) {
            fprintf(stderr, "spectral: '%s' ended early\n", path);
        }
    } while (0);

    fclose(file);
    return ok;
}

bool pt_spectral_init(gpu_device_t *device, gpu_uploader_t *uploader, const char *dir)
{
    if (g_loaded) {
        return true;
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/spectral.bin", dir);

    float *floats = gpu_alloc(PT_SPECTRAL_TOTAL_FLOATS * sizeof(float));
    const bool ok = read_blob(path, floats);
    if (!ok) {
        fprintf(stderr, "spectral: falling back to an achromatic table; colours will be grey\n");
        fill_achromatic(floats);
    }

    memcpy(g_cmf, &floats[PT_SPECTRAL_CMF_OFFSET], sizeof(g_cmf));

    // Needs the colour matching functions above, so it is computed after they are in place.
    // Without it the sky's sun lobe would be normalised at a single wavelength rather than by
    // luminance, which overstates it by the whole integral of ybar -- a factor of about 107.
    if (ok) {
        floats[14] = pt_spectral_blackbody_norm(PT_SPECTRAL_SUN_KELVIN);
    }

    const VkDeviceSize size = PT_SPECTRAL_TOTAL_FLOATS * sizeof(float);
    g_buffer = gpu_buffer_create(device, size,
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 GPU_MEMORY_DEVICE);
    gpu_buffer_upload(uploader, &g_buffer, floats, size);

    free(floats);
    g_loaded = true;
    return ok;
}

void pt_spectral_free(gpu_device_t *device)
{
    if (!g_loaded) {
        return;
    }
    gpu_buffer_destroy(device, &g_buffer);
    memset(&g_buffer, 0, sizeof(g_buffer));
    g_loaded = false;
}

VkDeviceAddress pt_spectral_address(void)
{
    return g_buffer.address;
}

float pt_spectral_blackbody_norm(float kelvin)
{
    if (kelvin <= 0.0f) {
        return 1.0f; // no temperature authored: the light keeps the colour it was given
    }

    // Planck's law with the 2hc^2 prefactor dropped -- this is a ratio, so any constant
    // multiplying the whole spectrum cancels. c2 = hc/k = 1.4387769e7 nm.K, the CIE second
    // radiation constant. The same shape the shader evaluates, for the same reason.
    double luminance = 0.0;
    for (uint32_t i = 0; i < PT_SPECTRAL_LAMBDA_COUNT; ++i) {
        const double lambda = (double)PT_SPECTRAL_LAMBDA_MIN + (double)i;
        const double fifth = lambda * lambda * lambda * lambda * lambda;
        const double shape = 1.0 / (fifth * (exp(1.4387769e7 / (lambda * (double)kelvin)) - 1.0));

        luminance += shape * (double)g_cmf[i * 3 + 1];
    }

    // The daylight basis in the blob already carries the normalisation that puts D65 at
    // Y = 1, and the shader's resolve is a plain sum against the colour matching functions
    // with no trailing constant. So matching D65's brightness means making this spectrum's
    // own sum against ybar equal 1 -- nothing else.
    //
    // A zero result means those functions never loaded, which the achromatic fallback leaves
    // at zero. Returning 1 keeps the light at the brightness it was authored with rather than
    // dividing by nothing.
    return luminance > 0.0 ? (float)(1.0 / luminance) : 1.0f;
}
