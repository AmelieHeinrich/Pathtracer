// A progressive path tracer over VK_KHR_ray_tracing_pipeline.
//
// Compiled as a single SPIR-V module holding every entry point, which requires
// slangc -fvk-use-entrypoint-name (Slang otherwise renames them all to "main").
// -fvk-use-scalar-layout makes these structs match the C ones byte for byte.
//
// The integrator is an iterative loop in raygen: closest hit shaders only describe the
// surface they landed on, and raygen owns the RNG, samples the BSDF and re-traces. Nothing
// traces from inside a hit shader, so maxPipelineRayRecursionDepth stays 1.

#include "primitives.slangh"
#include "spectral.slangh"

// Matches pt_vertex_t in src/scene.h.
struct Vertex {
    float3 position;
    float3 normal;
};

// Matches pt_instance_data_t in src/scene.h. Indexed by InstanceIndex(), so it runs
// index for index with the TLAS instance array.
struct InstanceData {
    Vertex *vertices; // null for the procedural shapes
    uint *indices;    // null for the procedural shapes
    float3 albedo;
    float3 emission;
    float roughness;
    float metallic;
    float transmission;
    float ior;
    float abbe;       // 0 disables dispersion
    // Beer-Lambert: the fraction surviving one unit of travel, already folded from the
    // authored colour and distance by fill_instance.
    float3 attenuation;
};

// Matches pt_camera_t in src/pt_math.h.
struct Camera {
    float3 position;
    float3 forward;
    float3 right;
    float3 up;
    float tan_half_fov;
    // Thin lens. A zero aperture is a pinhole and skips the lens sample entirely, so a scene
    // that does not ask for depth of field pays nothing for it and traces the very same rays
    // it did before these two existed.
    float aperture;
    float focus_distance;
};

// `InstanceData*` lowers to a SPIR-V PhysicalStorageBuffer pointer: a real buffer device
// address, not a descriptor. The scene can grow without touching the descriptor set.
// Matches pt_settings_t in src/renderer.h.
struct Settings {
    uint max_bounces;
    uint samples_per_frame;
    uint unlit;
    float sky_intensity; // 0 turns the sky off, leaving only the explicit lights
    float turbidity;
    float sun_azimuth;   // degrees
    float sun_elevation; // degrees
    // The sun's angular diameter in degrees. 0.53 is the real one; larger softens its
    // shadows. See sun_radiance -- this changes shadow softness, not scene brightness.
    float sun_angular_diameter;
};

// Matches pt_light_type_t in src/scene.h.
static const uint LIGHT_POINT = 0u;
static const uint LIGHT_DIRECTIONAL = 1u;
static const uint LIGHT_SPOT = 2u;
static const uint LIGHT_AREA = 3u;

// Matches pt_light_data_t in src/scene.h. None of these has geometry in the acceleration
// structure, so a BSDF ray can never land on one -- which is exactly why the explicit
// sampling below can add their contribution without double counting anything and without
// needing MIS. The area light is the interesting case: it is *drawn* for the camera ray only
// (see intersect_area_lights), so the panel is visible without a bounce ray ever finding it.
//
// A shape that glows is a different thing: an ordinary instance with a non-zero emission,
// found by BSDF rays alone.
struct Light {
    uint type;
    float3 position;
    float3 direction; // the direction the light points, already normalised
    float3 color;     // already multiplied by intensity
    float range;      // 0 = unbounded
    float cos_inner;  // cosines, so this shader never calls a trig function.
    float cos_outer;  // a larger cosine is a smaller angle, so cos_inner >= cos_outer
    float temperature; // Kelvin, or 0 for the default illuminant
    // Area: the rectangle's half-edge vectors and its full area, both built on the CPU.
    float3 edge_u;
    float3 edge_v;
    float area;
};

struct PushConstants {
    InstanceData *instances;
    Light *lights;
    // The baked spectral tables: colour matching functions, the daylight basis, and the
    // Jakob-Hanika upsampling LUT, all in one buffer. Layout in src/spectral.h, offsets in
    // spectral.slangh. A device address rather than a descriptor for the same reason the two
    // above are -- and because a 3D texture would need a sampler the GPU layer cannot make.
    float *spectra;
    Camera camera;
    // No aspect: raygen derives it from DispatchRaysDimensions, which is the extent the CPU
    // would have computed it from anyway. The four bytes are worth more than the convenience --
    // this block sits exactly on the 128 byte guarantee.
    uint frame_index;
    uint light_count;
    Settings settings;
};
[[vk::push_constant]] PushConstants push;

[[vk::binding(0, 0)]] uniform RaytracingAccelerationStructure scene;
[[vk::binding(1, 0)]] uniform RWTexture2D<float4> output;
// Running sum of radiance, never cleared: the frame_index == 0 test below stands in for a
// clear, and the image survives across frames while the camera holds still.
[[vk::binding(2, 0)]] uniform RWTexture2D<float4> accum;
// Camera space depth of the primary hit -- its distance along the camera's forward axis, not
// along the ray -- or PT_DEPTH_MISS where the ray escaped. Only the debug line pass reads it,
// to hide gizmos behind scene geometry. Depth along the axis rather than along the ray
// because that is what a rasterised line's own interpolated w is, and the two have to be the
// same quantity to be comparable: they differ by the cosine off the axis, which reaches 13%
// at the corners of a 40 degree view.
[[vk::binding(3, 0)]] uniform RWTexture2D<float> depth;
// The instance the primary ray hit, plus one, so 0 can mean "nothing". Read back a pixel at a
// time to turn a click into a selection -- which means picking runs through the very same
// intersection shaders the image does, and can never disagree with what is on screen.
[[vk::binding(4, 0)]] uniform RWTexture2D<uint> pick;
// World space normal at the primary hit, already oriented against the ray. Written for the
// denoiser alone: it is the one edge-stopping term that separates two faces meeting at a
// crease, where the depth is continuous and the instance is the same on both sides.
[[vk::binding(5, 0)]] uniform RWTexture2D<float4> normal_buffer;
// Two 128x128 blue noise tiles side by side, eight independent channels in all. Read only,
// uploaded once, and never resized -- see src/bluenoise.c.
[[vk::binding(6, 0)]] uniform RWTexture2D<float4> bluenoise;

// Larger than any TMax the integrator uses, so a line against the sky is never discarded.
static const float PT_DEPTH_MISS = 1e30f;

// Surface description only; no radiance and no recursion. hit_t < 0 marks a miss, in which
// case `emission` carries the sky instead.
struct Payload {
    float3 albedo;
    float3 emission;
    float3 normal; // world space, pointing outward -- raygen orients it against the ray
    float hit_t;
    float roughness;
    // Carried out of the hit shader because InstanceIndex() only exists there, and raygen is
    // where the pick image is written.
    uint instance;
};

// Shadow rays carry no surface description: they only need to know whether anything at all
// was in the way, which is why they skip the closest hit shaders entirely and let the miss
// shader be the one thing that runs on success.
struct ShadowPayload {
    bool visible;
    // Set by the any-hit when the ray passed *through* transmissive geometry on its way to the
    // light. Next event estimation through a refractive surface assumes the light is where it
    // appears in a straight line, which is true for a single interface -- a pane -- and false
    // for anything solid, where the light has to refract on the way in and again on the way
    // out. Aiming a sharp transmission lobe straight at a lamp that is not really there
    // overestimates by more than an order of magnitude, so the transmitted contribution is
    // dropped when this is set. The light still arrives; it arrives by the BSDF paths that
    // refract properly.
    bool crossed_transmissive;
};

// ---------------------------------------------------------------------------
// sampling
// ---------------------------------------------------------------------------

uint pcg_hash(uint value)
{
    const uint state = value * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct Rng {
    uint state;
};

float rand(inout Rng rng)
{
    rng.state = pcg_hash(rng.state);
    // Top 24 bits only, so every value is exactly representable in [0,1).
    return float(rng.state >> 8u) * (1.0f / 16777216.0f);
}

// ---------------------------------------------------------------------------
// blue noise
// ---------------------------------------------------------------------------

// Must match PT_BLUENOISE_TILE in src/bluenoise.h.
static const uint PT_BN_TILE = 128u;

// The R5 low-discrepancy sequence: alpha_d = frac(phi^-(d+1)) for phi the real root of
// x^6 = x + 1. Stored as 32 bit fixed point, which is what makes the multiply below exact --
// see blue_noise.
static const uint PT_BN_ALPHA[6] = {3785032106u, 3335640776u, 2939605022u,
                                    2590590015u, 2283013049u, 2011954246u};

// One sample of the low-dimensional prefix, after Heitz and Belcour's "distributing Monte
// Carlo errors as a blue noise distribution".
//
// A per-pixel blue noise value offsets a low-discrepancy sequence advanced by the sample
// index. Each pixel therefore gets a well-stratified sequence of its own, while at any fixed
// sample count the values *across* the image are blue noise distributed -- which is what makes
// the residual error land at high spatial frequencies, where both the eye and the a-trous
// filter deal with it best.
//
// This beats cycling through a stack of tiles: it needs one tile, and it keeps working at any
// sample count rather than repeating once the stack runs out.
float blue_noise(uint2 pixel, uint dim, uint sample_index)
{
    // Four channels to a tile, so the dimension picks both.
    const uint tile = dim >> 2u;
    const uint channel = dim & 3u;
    const uint2 coord = uint2((pixel.x % PT_BN_TILE) + tile * PT_BN_TILE, pixel.y % PT_BN_TILE);
    // Loaded into a local before the channel is selected. Indexing the load expression
    // directly makes Slang try to form a pointer into the image, which SPIR-V does not allow.
    const float4 texel = bluenoise[coord];
    const float base = texel[channel];

    // Fixed point on purpose. sample_index * alpha in floating point loses its low bits once
    // the accumulation has run for a while, and it is exactly those bits that carry the
    // sequence -- the stratification would quietly decay into a constant offset. An unsigned
    // multiply wraps modulo 2^32, which *is* the fractional part, exactly, forever.
    const uint offset = sample_index * PT_BN_ALPHA[dim];
    return frac(base + float(offset) * (1.0f / 4294967296.0f));
}

// A uniform point on the unit disc, by Shirley's concentric mapping. Concentric rather than
// the obvious sqrt(r), theta polar transform because that one compresses the square's corners
// into a thin wedge: it is still uniform, but it distorts the stratification of whatever
// sequence feeds it, which matters precisely when the sequence is a good one.
float2 concentric_disc(float u1, float u2)
{
    // To [-1, 1]^2. The exact centre would divide by zero below.
    const float2 offset = float2(u1, u2) * 2.0f - 1.0f;
    if (offset.x == 0.0f && offset.y == 0.0f) {
        return float2(0.0f);
    }

    // Written out rather than as PT_PI/4: bsdf.slangh, which defines PT_PI, is included
    // further down than this.
    const float quarter_pi = 0.785398163397448f;
    const float half_pi = 1.570796326794897f;

    float r;
    float theta;
    if (abs(offset.x) > abs(offset.y)) {
        r = offset.x;
        theta = quarter_pi * (offset.y / offset.x);
    } else {
        r = offset.y;
        theta = half_pi - quarter_pi * (offset.x / offset.y);
    }
    return r * float2(cos(theta), sin(theta));
}

// cosine_hemisphere used to live here. The diffuse lobe was inlined into bsdf_sample when the
// BSDF gained a proper lobe-selection pdf, leaving this with no callers -- and an uncalled
// sampling routine is worse than none, because it reads like a live source of random draws.

// The surface model. Included here rather than at the top because it builds on Rng and rand
// above, and on Spectrum from spectral.slangh.
#include "bsdf.slangh"

// ---------------------------------------------------------------------------
// direct lighting
// ---------------------------------------------------------------------------

// The light's colour and intensity as a spectrum. The authored colour is treated as a
// reflectance under D65, which is precisely what makes it resolve back to the colour that was
// authored -- so every scene written before the spectral port keeps its exact appearance.
Spectrum light_spectrum(Light light, float4 lambda)
{
    // A temperature swaps the illuminant the colour is lit by, rather than replacing the
    // colour -- so a warm bulb behind a blue gel still means something. fill_light has
    // already folded in the scale that holds the blackbody's luminance at 1, so the two
    // branches are the same brightness and only differ in hue.
    const Spectrum illuminant = light.temperature > 0.0f
                                    ? planck_shape(lambda, light.temperature)
                                    : illuminant_d65(push.spectra, lambda);
    return upsample_emission(push.spectra, light.color, illuminant, lambda);
}

// Works out how a single light reaches `hit_point`, before any ray is traced. Returns false
// when it cannot contribute at all -- out of range, below the horizon, or outside the spot
// cone -- so the caller can skip the shadow ray entirely.
// `two_sided` lets a light behind the surface through. A transmissive material can be lit from
// the far side, and rejecting those here would mean frosted glass could never see a lamp
// behind it -- the explicit lights are not in the acceleration structure, so this is their only
// route. For an opaque surface it stays false and the below-the-horizon early-out is kept,
// because a shadow ray toward a light that cannot contribute is pure waste.
bool sample_light(Light light, float4 lambda, float3 hit_point, float3 normal, bool two_sided,
                  inout Rng rng, out float3 to_light, out float distance_to_light,
                  out Spectrum incident)
{
    to_light = float3(0.0f);
    distance_to_light = 0.0f;
    incident = Spectrum(0.0f);

    const Spectrum emitted = light_spectrum(light, lambda);

    if (light.type == LIGHT_AREA) {
        // Uniform over the rectangle. The pdf is 1/area in *area* measure, and converting it
        // to solid angle divides by cos_light/d^2 -- so dividing the radiance by the pdf is
        // the same as multiplying it by area * cos_light / d^2, which is what happens below.
        // Named for the same reason the pixel jitter is: two operands of one expression have
        // no guaranteed evaluation order, and u must stay u.
        const float u = rand(rng);
        const float v = rand(rng);
        const float3 point = light.position + light.edge_u * (u * 2.0f - 1.0f) +
                             light.edge_v * (v * 2.0f - 1.0f);

        const float3 offset = point - hit_point;
        distance_to_light = length(offset);
        if (distance_to_light < 1e-5f) {
            return false;
        }
        to_light = offset / distance_to_light;

        // One sided: it emits along `direction` and is black from behind, which is what makes
        // a ceiling panel light the room and not the roof.
        const float cos_light = dot(-to_light, light.direction);
        if (cos_light <= 0.0f) {
            return false;
        }
        if (light.range > 0.0f && distance_to_light > light.range) {
            return false;
        }

        incident = emitted * (light.area * cos_light /
                              max(distance_to_light * distance_to_light, 1e-4f));
        return two_sided || dot(normal, to_light) > 0.0f;
    }

    if (light.type == LIGHT_DIRECTIONAL) {
        to_light = -light.direction;
        // Infinitely far away in principle; this just has to outrun the scene so the shadow
        // ray is not cut short before reaching an occluder.
        distance_to_light = 10000.0f;
        incident = emitted;
    } else {
        const float3 offset = light.position - hit_point;
        distance_to_light = length(offset);
        if (distance_to_light < 1e-5f) {
            return false;
        }
        to_light = offset / distance_to_light;

        if (light.range > 0.0f && distance_to_light > light.range) {
            return false;
        }

        // Inverse square, with the denominator floored: a light resting almost exactly on a
        // surface would otherwise produce a spike no amount of accumulation can average out.
        incident = emitted / max(distance_to_light * distance_to_light, 1e-4f);

        if (light.type == LIGHT_SPOT) {
            const float cosine = dot(-to_light, light.direction);
            if (cosine <= light.cos_outer) {
                return false;
            }
            // Smooth falloff from the outer cone in to the inner one, squared so the edge
            // reads as soft rather than as a hard ring.
            const float t = saturate((cosine - light.cos_outer) /
                                     max(light.cos_inner - light.cos_outer, 1e-4f));
            incident *= t * t;
        }
    }

    return two_sided || dot(normal, to_light) > 0.0f;
}

// One uniformly chosen light's contribution at a surface, shadow ray included. `throughput`
// is the path throughput *before* this surface's albedo has been folded in, because the
// BRDF below carries that albedo itself.
Spectrum direct_lighting(float3 hit_point, Surface surface, float3 wo, Spectrum throughput,
                         float4 lambda, inout Rng rng)
{
    if (push.light_count == 0u) {
        return Spectrum(0.0f);
    }

    const uint index = min(uint(rand(rng) * float(push.light_count)), push.light_count - 1u);
    const Light light = push.lights[index];

    float3 to_light;
    float distance_to_light;
    Spectrum incident;
    if (!sample_light(light, lambda, hit_point, surface.normal, surface.transmission > 0.0f,
                      rng, to_light, distance_to_light, incident)) {
        return Spectrum(0.0f);
    }

    RayDesc shadow;
    // Offset toward whichever side the light is on. A transmissive surface can be lit from
    // behind, and pushing that ray out along the front-facing normal would start it inside the
    // very surface it is leaving.
    shadow.Origin = hit_point + (dot(surface.normal, to_light) > 0.0f ? surface.normal
                                                                     : -surface.normal) *
                                    0.0005f;
    shadow.Direction = to_light;
    shadow.TMin = 0.001f;
    // Stops short of the light itself, so a surface is never shadowed by the thing lighting it.
    shadow.TMax = distance_to_light - 0.001f;

    ShadowPayload occlusion;
    occlusion.visible = false;
    occlusion.crossed_transmissive = false;
    // Miss shader 1, and no closest hit shader at all: the first hit of any kind ends the
    // search, and only reaching the miss shader means the light is visible.
    // FORCE_NON_OPAQUE is what makes the any-hit above run at all: the acceleration structure
    // is built with the opaque flag, which otherwise skips it. Asking for it here and nowhere
    // else is what keeps the cost on the one ray type that needs it.
    TraceRay(scene,
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER |
                 RAY_FLAG_FORCE_NON_OPAQUE,
             0xFFu, 0, 0, 1, shadow, occlusion);

    if (!occlusion.visible) {
        return Spectrum(0.0f);
    }

    // A light on the far side reaches this surface only through its transmission lobe, and
    // that lobe's straight-line assumption only holds across a single interface. If the ray
    // had to cross other glass to get here, it is not the path light actually takes.
    if (occlusion.crossed_transmissive && dot(surface.normal, to_light) < 0.0f) {
        return Spectrum(0.0f);
    }

    // A real evaluated BSDF, including the cosine. It returns zero for a delta lobe, which is
    // what keeps a punctual light off a mirror -- the `* roughness` scaling that used to stand
    // in for that is gone, and with it the reason roughness had to mean two things at once.
    //
    // Reflectance and illuminant multiply wavelength by wavelength here and are only resolved
    // at the end of the path, which is what makes colour bleeding pick up the tints it
    // physically should.
    const Spectrum f = bsdf_eval(surface, wo, to_light);
    return throughput * f * incident * float(push.light_count);
}

// The sun, sampled explicitly, MIS-weighted against the BSDF's own chance of finding it.
//
// This is what makes a physically sized sun affordable. A half-degree disc covers about 7e-5
// steradians: a cosine-weighted bounce ray lands inside it roughly once in fifty thousand
// tries, and on the rare occasion it does the sample carries the entire sun, which is a
// firefly rather than sunlight. Aiming at the disc turns that into one well-behaved sample per
// bounce -- and the MIS weight is what keeps the two routes from double counting.
//
// The dome is deliberately not sampled here. It is broad and low-contrast, BSDF sampling
// handles it perfectly well, and a dome CDF would have to track the Preetham fit including its
// ground term for very little gain.
Spectrum sun_lighting(float3 hit_point, Surface surface, float3 wo, Spectrum throughput,
                      float4 lambda, inout Rng rng)
{
    if (push.settings.sky_intensity <= 0.0f) {
        return Spectrum(0.0f);
    }

    const float3 sun_dir = sun_direction();
    const float solid_angle = sun_solid_angle();
    const float half_angle = radians(max(push.settings.sun_angular_diameter, 0.01f) * 0.5f);
    const float cos_half = cos(half_angle);

    // Uniform inside the cone. The two draws are named for the usual reason.
    const float u1 = rand(rng);
    const float u2 = rand(rng);
    const float cos_theta = 1.0f - u1 * (1.0f - cos_half);
    const float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
    const float phi = 6.28318530718f * u2;

    const Frame sun_frame = frame_from_normal(sun_dir);
    const float3 to_light = to_world(sun_frame, float3(sin_theta * cos(phi),
                                                       sin_theta * sin(phi), cos_theta));

    // Behind the surface, and not reachable through it either.
    const bool two_sided = surface.transmission > 0.0f;
    if (!two_sided && dot(surface.normal, to_light) <= 0.0f) {
        return Spectrum(0.0f);
    }

    const Spectrum f = bsdf_eval(surface, wo, to_light);
    if (spectrum_max(f) <= 0.0f) {
        return Spectrum(0.0f); // a delta lobe, or the wrong side; nothing to weight
    }

    RayDesc shadow;
    shadow.Origin = hit_point + (dot(surface.normal, to_light) > 0.0f ? surface.normal
                                                                     : -surface.normal) *
                                    0.0005f;
    shadow.Direction = to_light;
    shadow.TMin = 0.001f;
    // The sun is infinitely far away, so this ray only stops when it leaves the scene.
    shadow.TMax = 10000.0f;

    ShadowPayload occlusion;
    occlusion.visible = false;
    occlusion.crossed_transmissive = false;
    TraceRay(scene,
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER |
                 RAY_FLAG_FORCE_NON_OPAQUE,
             0xFFu, 0, 0, 1, shadow, occlusion);

    if (!occlusion.visible) {
        return Spectrum(0.0f);
    }
    // The same straight-line caveat direct_lighting documents: a transmission lobe aimed
    // through other glass is not the path the light actually takes.
    if (occlusion.crossed_transmissive && dot(surface.normal, to_light) < 0.0f) {
        return Spectrum(0.0f);
    }

    // The power heuristic. Squaring is what makes it prefer the better-matched strategy more
    // sharply than the balance heuristic would, which is the usual choice and the one that
    // handles a near-specular lobe against a small light best.
    const float pdf_sun = 1.0f / solid_angle;
    const float pdf_bsdf = bsdf_pdf(surface, wo, to_light);
    const float weight = (pdf_sun * pdf_sun) /
                         (pdf_sun * pdf_sun + pdf_bsdf * pdf_bsdf);

    return throughput * f * sun_radiance(to_light, lambda) * (weight / pdf_sun);
}

// Nearest area light rectangle along a ray, if any. Area lights are not in the acceleration
// structure -- which is exactly what lets direct_lighting sample them without double counting
// anything -- so seeing the panel at all takes this explicit test.
//
// Only ever called for the camera ray. On a bounce ray the same light has already been
// accounted for by next event estimation at the previous surface, and adding it here as well
// would count it twice.
bool intersect_area_lights(RayDesc ray, float4 lambda, out float hit_t, out Spectrum emission)
{
    hit_t = ray.TMax;
    emission = Spectrum(0.0f);
    bool found = false;

    for (uint i = 0u; i < push.light_count; ++i) {
        const Light light = push.lights[i];
        if (light.type != LIGHT_AREA) {
            continue;
        }

        const float denominator = dot(ray.Direction, light.direction);
        // One sided, matching the sampling above: only the emitting face is visible.
        if (denominator >= -1e-6f) {
            continue;
        }

        const float t = dot(light.position - ray.Origin, light.direction) / denominator;
        if (t < ray.TMin || t > hit_t) {
            continue;
        }

        // Inside the rectangle? The half-edges are orthogonal, so projecting onto each and
        // comparing against its own squared length is the whole test.
        const float3 offset = ray.Origin + ray.Direction * t - light.position;
        const float u = dot(offset, light.edge_u);
        const float v = dot(offset, light.edge_v);
        if (abs(u) > dot(light.edge_u, light.edge_u) ||
            abs(v) > dot(light.edge_v, light.edge_v)) {
            continue;
        }

        hit_t = t;
        emission = light_spectrum(light, lambda);
        found = true;
    }

    return found;
}

// ---------------------------------------------------------------------------
// sky
// ---------------------------------------------------------------------------

// Preetham's luminance arrives in kcd/m^2, running roughly 1 to 20 for a daylit sky, while
// this renderer has no exposure control and its scenes are authored around values near 1.
// This is the constant that reconciles the two. It is empirical and unashamedly so -- there
// is no tonemapper to hide a mismatch behind, so it is written here rather than buried.
static const float PT_SKY_SCALE = 0.05f;
// The sun lobe's brightness, in the same units, chosen to sit a little above the sky it is
// seen against.
static const float PT_SUN_INTENSITY = 8.0f;

// The sky along a ray, as a spectrum. Evaluated here rather than in the miss shader so the
// payload never has to carry wavelengths in and a spectrum back out.
// The sun's direction, from the two settings the panel exposes. Azimuth is measured
// clockwise from -Z and elevation up from the horizon.
//
// Hoisted out of sky_radiance because three separate places need it now: the dome, the disc,
// and the explicit sun sampling that makes the disc affordable.
float3 sun_direction()
{
    const float azimuth = radians(push.settings.sun_azimuth);
    const float elevation = radians(push.settings.sun_elevation);
    return normalize(float3(sin(azimuth) * cos(elevation), sin(elevation),
                            -cos(azimuth) * cos(elevation)));
}

// The solid angle the sun's disc covers, in steradians.
float sun_solid_angle()
{
    const float half_angle = radians(max(push.settings.sun_angular_diameter, 0.01f) * 0.5f);
    return 2.0f * PT_PI * (1.0f - cos(half_angle));
}

// The Preetham dome alone, with no disc in it. Split from the disc because MIS has to weight
// the two differently: the dome is reachable only by BSDF sampling, while the disc is
// reachable both ways and would otherwise be counted twice.
Spectrum sky_dome_radiance(float3 dir, float4 lambda)
{
    if (push.settings.sky_intensity <= 0.0f) {
        return Spectrum(0.0f);
    }
    return preetham_sky(push.spectra, dir, sun_direction(), push.settings.turbidity,
                        PT_SKY_SCALE, lambda) *
           push.settings.sky_intensity;
}

// The sun's disc: a 5778 K blackbody inside a cone of the authored angular diameter, and
// nothing outside it.
//
// The radiance is divided by the disc's solid angle, which is what makes the *illuminance*
// independent of how wide the disc is: narrowing the sun sharpens its shadows without
// darkening the scene, so the angular diameter reads as a softness control rather than as a
// second brightness control. The 2*PI/51 factor is the integral of the pow(cos, 50) lobe this
// replaces, so PT_SUN_INTENSITY still means what it meant before.
//
// A physically sized disc is only affordable because sun_lighting samples it explicitly. Found
// by chance alone -- which is all the renderer could do before -- a half-degree sun arrives as
// fireflies rather than as sunlight.
Spectrum sun_radiance(float3 dir, float4 lambda)
{
    if (push.settings.sky_intensity <= 0.0f) {
        return Spectrum(0.0f);
    }

    const float half_angle = radians(max(push.settings.sun_angular_diameter, 0.01f) * 0.5f);
    if (dot(dir, sun_direction()) < cos(half_angle)) {
        return Spectrum(0.0f);
    }

    // Scaled to luminance 1 by the baked normalisation, so PT_SUN_INTENSITY reads as a plain
    // brightness. Normalising at a single wavelength instead would overstate it by the whole
    // integral of ybar -- a factor of about 107 -- because every other spectrum here carries
    // its normalisation as a luminance, not as a value at one wavelength.
    const float scale = PT_SUN_INTENSITY * (2.0f * PT_PI / 51.0f) / sun_solid_angle();
    return planck_shape(lambda, PT_SUN_KELVIN) *
           (scale * push.spectra[PT_OFF_SUN_NORM] * push.settings.sky_intensity);
}

// Everything the sky sends along `dir`. Only the unlit view uses this now; the integrator
// wants the two halves separately so it can weight them.
Spectrum sky_radiance(float3 dir, float4 lambda)
{
    return sky_dome_radiance(dir, lambda) + sun_radiance(dir, lambda);
}

// ---------------------------------------------------------------------------
// entry points
// ---------------------------------------------------------------------------

// One complete path through the scene, from a jittered point inside `pixel`. The caller
// takes as many of these per frame as push.settings.samples_per_frame asks for; `rng` is
// carried across them so successive samples do not repeat one another.
float3 trace_sample(uint2 pixel, uint2 size, uint sample_index, inout Rng rng,
                    out float primary_t, out uint primary_instance, out float3 primary_normal)
{
    primary_t = PT_DEPTH_MISS;
    primary_instance = 0u; // 0 means nothing was hit
    primary_normal = float3(0.0f, 1.0f, 0.0f); // replaced below on the first bounce

    // One hero wavelength for the whole path, drawn before the first ray. Every spectral
    // quantity below is sampled at exactly these four, and they are resolved to a colour once,
    // at the very end -- never per bounce.
    // Dimension 0. The hero wavelength decides the whole path's spectrum, so stratifying it
    // across the image is what stops the spectral noise from clumping into colour blotches.
    const float4 lambda = hero_wavelengths(blue_noise(pixel, 0u, sample_index));
    // Not const: a dispersive refraction narrows this to the hero wavelength alone.
    float4 lambda_pdf = float4(PT_LAMBDA_PDF);

    // Sub-pixel jitter: free antialiasing, since the frames are being averaged anyway.
    //
    // Dimensions 1 and 2. Named rather than written inline, because evaluation order within
    // an expression is unspecified and these two are not interchangeable.
    const float jitter_x = blue_noise(pixel, 1u, sample_index);
    const float jitter_y = blue_noise(pixel, 2u, sample_index);
    float2 ndc = (float2(pixel) + float2(jitter_x, jitter_y)) / float2(size) * 2.0f - 1.0f;
    // The aspect ratio is the dispatch's own, which is what the CPU used to send down.
    ndc.x *= float(size.x) / float(size.y);

    const Camera camera = push.camera;

    // The unnormalised direction, kept before normalising because its component along
    // `forward` is exactly 1: the point camera.position + direction therefore lies on the
    // plane one unit ahead, and scaling it by focus_distance lands on the focal plane
    // directly, with no trigonometry and no division by a cosine.
    const float3 direction = camera.forward + (ndc.x * camera.tan_half_fov) * camera.right -
                             (ndc.y * camera.tan_half_fov) * camera.up;

    RayDesc ray;
    ray.Origin = camera.position;
    ray.Direction = normalize(direction);

    if (camera.aperture > 0.0f) {
        // Thin lens: every ray through this pixel has to pass through the same focal point,
        // so the plane at focus_distance stays sharp while everything else is smeared across
        // the aperture. The draws are named for the usual unspecified-order reason.
        // Dimensions 3 and 4. The lens is where blue noise pays off most visibly: a bokeh
        // disc is a wide, low-frequency blur, so unstratified lens samples read as mottling
        // across the whole out-of-focus region rather than as fine grain.
        const float lens_u = blue_noise(pixel, 3u, sample_index);
        const float lens_v = blue_noise(pixel, 4u, sample_index);
        const float2 lens = concentric_disc(lens_u, lens_v) * camera.aperture;

        const float3 focal_point = camera.position + direction * camera.focus_distance;
        ray.Origin = camera.position + camera.right * lens.x + camera.up * lens.y;
        ray.Direction = normalize(focal_point - ray.Origin);
    }

    ray.TMin = 0.001f;
    ray.TMax = 10000.0f;

    Spectrum radiance = Spectrum(0.0f);
    Spectrum throughput = Spectrum(1.0f);

    // What produced the ray currently being traced, carried across iterations so the sky hit
    // below can be MIS-weighted against the sun sampling that could also have found it.
    // A camera ray had no preceding sun sample, so it starts as a delta: weight 1.
    float prev_bsdf_pdf = 0.0f;
    bool prev_specular = true;

    // Beer-Lambert state. One medium rather than a stack, which is exactly right as long as
    // transmissive objects do not overlap -- nest two and the inner one's exit clears the
    // outer one's absorption. `inside` is tracked separately from the value so that a
    // perfectly clear material still counts as being entered.
    Spectrum medium = Spectrum(1.0f);
    bool inside = false;

    for (uint bounce = 0u; bounce < push.settings.max_bounces; ++bounce) {
        Payload payload;
        payload.hit_t = -1.0f;
        payload.albedo = float3(0.0f);
        payload.emission = float3(0.0f);
        payload.normal = float3(0.0f, 1.0f, 0.0f);
        payload.roughness = 1.0f;
        payload.instance = 0u;

        // The instance's SBT record offset alone picks the hit group, so the ray
        // contribution and the geometry multiplier both stay 0.
        TraceRay(scene, RAY_FLAG_NONE, 0xFFu, 0, 0, 0, ray, payload);

        // Absorption applies to the segment the ray just finished, so it is folded in here,
        // before anything that segment reached gets added. Raising the per-unit transmittance
        // to the distance travelled *is* Beer-Lambert -- and doing it per wavelength is what
        // makes a thick piece of coloured glass shift hue rather than just darken.
        if (inside && payload.hit_t > 0.0f) {
            throughput *= pow(medium, Spectrum(payload.hit_t));
        }

        // An area light panel in front of the geometry, seen directly. Camera ray only: on a
        // bounce ray the same light was already accounted for by next event estimation at the
        // previous surface, and adding it again here would count it twice.
        if (bounce == 0u) {
            float light_t;
            Spectrum light_emission;
            if (intersect_area_lights(ray, lambda, light_t, light_emission) &&
                (payload.hit_t < 0.0f || light_t < payload.hit_t)) {
                // The panel is opaque, so this both ends the path and stands in for the hit.
                radiance += throughput * light_emission;
                primary_t = light_t * dot(ray.Direction, camera.forward);
                primary_instance = 0u; // not an entity, so nothing to select
                // intersect_area_lights does not hand back the panel's orientation, and
                // facing the camera is close enough for an edge stop: a flat emitter is the
                // one surface with nothing on it for the filter to smear.
                primary_normal = -ray.Direction;
                break;
            }
        }

        // Reported before the unlit early-out below, so the debug pass can occlude its lines
        // in every view mode.
        if (bounce == 0u) {
            // hit_t is along the ray; both vectors are unit, so this cosine turns it into
            // depth along the camera axis. See the binding 3 comment.
            primary_t = payload.hit_t < 0.0f
                            ? PT_DEPTH_MISS
                            : payload.hit_t * dot(ray.Direction, camera.forward);
            // One-based, so 0 stays available to mean "the ray escaped".
            primary_instance = payload.hit_t < 0.0f ? 0u : payload.instance + 1u;
            // Oriented against the ray, the same way the shading normal is below -- a back
            // face and a front face at the same crease must not read as opposite normals.
            //
            // A miss writes the negated ray direction rather than zero, which matters: a zero
            // vector would make dot(n0, n1) zero for every pair of sky pixels and switch the
            // filter off across the whole background. The negated direction varies smoothly
            // across the sky, so sky neighbours weight ~1, while sky against geometry is
            // already rejected by the depth term -- PT_DEPTH_MISS against any finite depth.
            primary_normal = payload.hit_t < 0.0f
                                 ? -ray.Direction
                                 : (dot(payload.normal, ray.Direction) < 0.0f
                                        ? payload.normal
                                        : -payload.normal);
        }

        // Unlit: the surface as the material describes it, with no light transport at all.
        // A miss still carries the sky in `emission`, so the background is unchanged.
        // Unlit: the surface as the material describes it, with no light transport at all.
        // Still routed through the spectral path rather than short circuited to RGB, which
        // makes this view a free and very sharp check on the upsampling: an albedo upsampled
        // under D65 and resolved back must give the colour that was authored.
        if (push.settings.unlit != 0u) {
            const Spectrum illuminant = illuminant_d65(push.spectra, lambda);
            radiance = payload.hit_t < 0.0f
                           ? sky_radiance(ray.Direction, lambda)
                           : upsample_reflectance(push.spectra, payload.albedo, lambda) *
                                     illuminant +
                                 upsample_emission(push.spectra, payload.emission, illuminant,
                                                   lambda);
            break;
        }

        // The ray escaped: the sky is the only thing left on this path.
        if (payload.hit_t < 0.0f) {
            // The dome at full weight -- only this route ever generates it, since sun_lighting
            // samples the disc alone.
            radiance += throughput * sky_dome_radiance(ray.Direction, lambda);

            // The disc, weighted against the sun sampling that could have found the same
            // direction. A camera ray and a delta bounce had no such sample to compete with,
            // so they take the whole thing; anything else splits it by the power heuristic,
            // and the two weights sum to one across the two strategies.
            const Spectrum disc = sun_radiance(ray.Direction, lambda);
            if (spectrum_max(disc) > 0.0f) {
                float weight = 1.0f;
                if (!prev_specular) {
                    const float pdf_sun = 1.0f / sun_solid_angle();
                    weight = (prev_bsdf_pdf * prev_bsdf_pdf) /
                             (prev_bsdf_pdf * prev_bsdf_pdf + pdf_sun * pdf_sun);
                }
                radiance += throughput * disc * weight;
            }
            break;
        }

        // A surface's own glow. An emitter, so it upsamples against the illuminant rather
        // than as a reflectance.
        radiance += throughput * upsample_emission(push.spectra, payload.emission,
                                                   illuminant_d65(push.spectra, lambda),
                                                   lambda);

        const float3 hit_point = ray.Origin + ray.Direction * payload.hit_t;
        const InstanceData material = push.instances[payload.instance];

        // The bit fill_surface deliberately preserved. Everything downstream wants a normal
        // facing the ray, so it is oriented here -- once, where the ray direction is in scope.
        const bool front_face = dot(payload.normal, ray.Direction) < 0.0f;
        const float3 normal = front_face ? payload.normal : -payload.normal;

        // The material fields the payload does not carry are read straight from the instance
        // table here. The payload already identifies the instance, so widening it to carry
        // them as well would cost every hit shader memory for nothing.
        const Surface surface =
            make_surface(upsample_reflectance(push.spectra, payload.albedo, lambda), normal,
                         payload.roughness, material.metallic, material.transmission,
                         cauchy_ior(material.ior, material.abbe, lambda), front_face);

        // Points back the way the ray came, which is the convention the BSDF is written in.
        const float3 wo = -ray.Direction;

        // Explicit light sampling, before the throughput is updated below -- the BSDF
        // evaluated inside carries the surface's own reflectance.
        radiance += direct_lighting(hit_point, surface, wo, throughput, lambda, rng);
        radiance += sun_lighting(hit_point, surface, wo, throughput, lambda, rng);

        const BsdfSample bsdf = bsdf_sample(surface, wo, rng);
        if (spectrum_max(bsdf.weight) <= 0.0f) {
            break; // the lobe sampled into the surface, so the path ends here
        }
        throughput *= bsdf.weight;

        // Recorded before the ray is replaced below, for the sky test at the top of the next
        // iteration. A delta lobe has no density for sun sampling to compete with, which is
        // exactly what BsdfSample::specular has always meant -- this is the first thing to
        // actually read it.
        prev_specular = bsdf.specular;
        prev_bsdf_pdf = bsdf.specular ? 0.0f : bsdf_pdf(surface, wo, bsdf.direction);

        // A dispersive refraction sends the four wavelengths four different ways, and a path
        // can only follow one. Drop the other three by zeroing their wavelength densities:
        // spectral_resolve divides by the *mean* of these, so three zeros turn it into the
        // correct single-wavelength estimator with nothing else to change. Once collapsed a
        // path stays collapsed, which is why this only ever narrows.
        if (bsdf.dispersive) {
            lambda_pdf = float4(lambda_pdf.x, 0.0f, 0.0f, 0.0f);
            throughput = Spectrum(throughput.x, 0.0f, 0.0f, 0.0f);
        }

        // Crossing the boundary is what starts and stops absorption. Which way depends only
        // on the face the ray arrived at: entering through the front, leaving through the back.
        if (bsdf.transmitted) {
            inside = front_face;
            medium = inside ? upsample_reflectance(push.spectra, material.attenuation, lambda)
                            : Spectrum(1.0f);
        }

        // A transmitted ray leaves on the far side, so the offset that keeps a reflection off
        // its own surface would push this one straight back into it.
        ray.Origin = hit_point + (bsdf.transmitted ? -normal : normal) * 0.0005f;
        ray.Direction = bsdf.direction;
        ray.TMin = 0.001f;
        ray.TMax = 10000.0f;

        // Russian roulette, once a path is long enough that the bias-free early exit is
        // worth the added variance.
        if (bounce >= 2u) {
            const float survive = clamp(spectrum_max(throughput), 0.02f, 1.0f);
            if (rand(rng) > survive) {
                break;
            }
            throughput /= survive;
        }
    }

    // The one place spectra become a colour. Resolving per path rather than accumulating
    // spectrally is not a shortcut: each path draws its own hero wavelength, so there is no
    // fixed set of wavelengths for the accumulation image to hold.
    return spectral_resolve(push.spectra, lambda, radiance, lambda_pdf);
}

[shader("raygeneration")]
void raygen()
{
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint2 size = DispatchRaysDimensions().xy;

    Rng rng;
    rng.state = pcg_hash(pixel.x + pixel.y * size.x + pcg_hash(push.frame_index));

    // Clamped rather than trusted: a zero would divide the accumulator by zero below.
    const uint samples = max(push.settings.samples_per_frame, 1u);

    float3 radiance = float3(0.0f);
    // Written from the first sample only: every sample lands in the same pixel, so any one of
    // them describes the geometry there, and the debug pass wants one number, not an average.
    float primary_t = PT_DEPTH_MISS;
    uint primary_instance = 0u;
    float3 primary_normal = float3(0.0f, 1.0f, 0.0f);
    for (uint sample = 0u; sample < samples; ++sample) {
        float sample_t;
        uint sample_instance;
        float3 sample_normal;
        // The global sample index, not the loop counter: the sequence has to keep advancing
        // across frames, which is the whole reason the accumulation converges.
        const uint sample_index = push.frame_index * samples + sample;
        radiance += trace_sample(pixel, size, sample_index, rng, sample_t, sample_instance,
                                 sample_normal);
        if (sample == 0u) {
            primary_t = sample_t;
            primary_instance = sample_instance;
            primary_normal = sample_normal;
        }
    }
    depth[pixel] = primary_t;
    pick[pixel] = primary_instance;
    normal_buffer[pixel] = float4(primary_normal, 0.0f);

    // Testing frame_index == 0 is what stands in for ever clearing the accumulation image.
    // The accumulator holds a sum, so the divisor is the total number of samples that went
    // into it, not the number of frames.
    const float3 total = push.frame_index == 0u ? radiance : accum[pixel].rgb + radiance;
    accum[pixel] = float4(total, 1.0f);
    output[pixel] = float4(total / float((push.frame_index + 1u) * samples), 1.0f);
}

[shader("miss")]
void miss(inout Payload payload)
{
    // Deliberately does nothing but record the miss. The sky is spectral now, and evaluating
    // it here would mean carrying four wavelengths in and a four wide spectrum back out --
    // eight more floats in a payload every hit shader also pays for. raygen already holds the
    // ray direction, so it evaluates the sky itself; see sky_radiance in trace_sample.
    payload.emission = float3(0.0f);
    payload.hit_t = -1.0f;
}

// Lets a shadow ray pass through transmissive geometry instead of being stopped by it.
//
// Without this a glass sphere casts a solid black shadow, which is the single most obvious
// artifact a transmissive material can have. Ignoring the hit outright is an approximation --
// the light is neither bent nor tinted on its way through, so there is no focused caustic in
// the shadow -- but the real caustic still arrives by the BSDF paths that refract properly,
// and a hard black disc under a glass ball is far more wrong than a missing focus.
//
// Only shadow rays ever run this. The geometry is built opaque, so an any-hit is skipped
// unless a ray asks for it, and only direct_lighting's does (RAY_FLAG_FORCE_NON_OPAQUE).
// Primary and bounce rays therefore pay nothing and still see glass normally.
//
// Two copies because the attribute type differs: the mesh group reports barycentrics and the
// procedural groups report their own struct. Neither body reads them.
[shader("anyhit")]
void anyhit_shadow_mesh(inout ShadowPayload payload,
                        BuiltInTriangleIntersectionAttributes attributes)
{
    if (push.instances[InstanceIndex()].transmission > 0.0f) {
        payload.crossed_transmissive = true;
        IgnoreHit();
    }
}

[shader("anyhit")]
void anyhit_shadow_procedural(inout ShadowPayload payload, ProcAttrib attributes)
{
    if (push.instances[InstanceIndex()].transmission > 0.0f) {
        payload.crossed_transmissive = true;
        IgnoreHit();
    }
}

// Reached only when a shadow ray got all the way to the light without hitting anything.
[shader("miss")]
void miss_shadow(inout ShadowPayload payload)
{
    payload.visible = true;
}

// Copies the material out of the instance table. The normal is left pointing *outward*,
// unflipped: which side of a surface the ray arrived on is exactly the bit that chooses
// between an index ratio and its reciprocal, so throwing it away here -- as this used to --
// would make refraction impossible. raygen orients it instead, and keeps the bit.
void fill_surface(inout Payload payload, float3 normal)
{
    const InstanceData instance = push.instances[InstanceIndex()];

    payload.albedo = instance.albedo;
    payload.emission = instance.emission;
    payload.roughness = instance.roughness;
    payload.hit_t = RayTCurrent();
    payload.normal = normal;
    payload.instance = InstanceIndex();
}

[shader("closesthit")]
void closesthit_mesh(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes)
{
    const InstanceData instance = push.instances[InstanceIndex()];

    // Hardware gives two barycentrics; the third is implied.
    const float2 bary = attributes.barycentrics;
    const float3 weights = float3(1.0f - bary.x - bary.y, bary.x, bary.y);

    const uint base = PrimitiveIndex() * 3u;
    const Vertex v0 = instance.vertices[instance.indices[base + 0u]];
    const Vertex v1 = instance.vertices[instance.indices[base + 1u]];
    const Vertex v2 = instance.vertices[instance.indices[base + 2u]];

    const float3 object_normal =
        weights.x * v0.normal + weights.y * v1.normal + weights.z * v2.normal;

    fill_surface(payload, normalize(mul(object_normal, (float3x3)WorldToObject3x4())));
}

// Shared by all three procedural hit groups: they differ only in their intersection shader.
[shader("closesthit")]
void closesthit_procedural(inout Payload payload, ProcAttrib attributes)
{
    // Multiplying the normal by the world-to-object matrix on the left is the
    // inverse transpose, which is what keeps normals correct under the non-uniform scales
    // the instance transforms apply.
    fill_surface(payload,
                 normalize(mul(attributes.object_normal, (float3x3)WorldToObject3x4())));
}
