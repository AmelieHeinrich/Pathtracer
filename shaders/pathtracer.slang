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
    float pad;
};

// Matches pt_camera_t in src/renderer.h.
struct Camera {
    float3 position;
    float3 forward;
    float3 right;
    float3 up;
    float tan_half_fov;
};

// `InstanceData*` lowers to a SPIR-V PhysicalStorageBuffer pointer: a real buffer device
// address, not a descriptor. The scene can grow without touching the descriptor set.
// Matches pt_settings_t in src/renderer.h.
struct Settings {
    uint max_bounces;
    uint samples_per_frame;
    uint unlit;
    float sky_intensity; // 0 turns the sky off, leaving only the explicit lights
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
    // Area: the rectangle's half-edge vectors and its full area, both built on the CPU.
    float3 edge_u;
    float3 edge_v;
    float area;
};

struct PushConstants {
    InstanceData *instances;
    Light *lights;
    Camera camera;
    float aspect;
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

// Larger than any TMax the integrator uses, so a line against the sky is never discarded.
static const float PT_DEPTH_MISS = 1e30f;

// Surface description only; no radiance and no recursion. hit_t < 0 marks a miss, in which
// case `emission` carries the sky instead.
struct Payload {
    float3 albedo;
    float3 emission;
    float3 normal; // world space, turned to face the incoming ray
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

// Cosine weighted around `n`. Pairing this pdf with a Lambertian BRDF makes the whole
// throughput update collapse to a single multiply by the albedo.
float3 cosine_hemisphere(float3 n, inout Rng rng)
{
    const float u1 = rand(rng);
    const float u2 = rand(rng);

    const float r = sqrt(u1);
    const float phi = 6.28318530718f * u2;

    // Any tangent will do; picking the axis `n` is least aligned with keeps the cross
    // product well conditioned.
    const float3 helper = abs(n.x) > 0.9f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    const float3 tangent = normalize(cross(n, helper));
    const float3 bitangent = cross(n, tangent);

    return normalize(tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) +
                     n * sqrt(max(0.0f, 1.0f - u1)));
}

// ---------------------------------------------------------------------------
// direct lighting
// ---------------------------------------------------------------------------

// Works out how a single light reaches `hit_point`, before any ray is traced. Returns false
// when it cannot contribute at all -- out of range, below the horizon, or outside the spot
// cone -- so the caller can skip the shadow ray entirely.
bool sample_light(Light light, float3 hit_point, float3 normal, inout Rng rng,
                  out float3 to_light, out float distance_to_light, out float3 incident)
{
    to_light = float3(0.0f);
    distance_to_light = 0.0f;
    incident = float3(0.0f);

    if (light.type == LIGHT_AREA) {
        // Uniform over the rectangle. The pdf is 1/area in *area* measure, and converting it
        // to solid angle divides by cos_light/d^2 -- so dividing the radiance by the pdf is
        // the same as multiplying it by area * cos_light / d^2, which is what happens below.
        const float3 point = light.position +
                             light.edge_u * (rand(rng) * 2.0f - 1.0f) +
                             light.edge_v * (rand(rng) * 2.0f - 1.0f);

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

        incident = light.color * (light.area * cos_light /
                                  max(distance_to_light * distance_to_light, 1e-4f));
        return dot(normal, to_light) > 0.0f;
    }

    if (light.type == LIGHT_DIRECTIONAL) {
        to_light = -light.direction;
        // Infinitely far away in principle; this just has to outrun the scene so the shadow
        // ray is not cut short before reaching an occluder.
        distance_to_light = 10000.0f;
        incident = light.color;
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
        incident = light.color / max(distance_to_light * distance_to_light, 1e-4f);

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

    return dot(normal, to_light) > 0.0f;
}

// One uniformly chosen light's contribution at a surface, shadow ray included. `throughput`
// is the path throughput *before* this surface's albedo has been folded in, because the
// BRDF below carries that albedo itself.
float3 direct_lighting(float3 hit_point, Payload payload, float3 throughput, inout Rng rng)
{
    if (push.light_count == 0u || payload.roughness <= 0.0f) {
        return float3(0.0f);
    }

    const uint index = min(uint(rand(rng) * float(push.light_count)), push.light_count - 1u);
    const Light light = push.lights[index];

    float3 to_light;
    float distance_to_light;
    float3 incident;
    if (!sample_light(light, hit_point, payload.normal, rng, to_light, distance_to_light,
                      incident)) {
        return float3(0.0f);
    }

    RayDesc shadow;
    shadow.Origin = hit_point + payload.normal * 0.0005f;
    shadow.Direction = to_light;
    shadow.TMin = 0.001f;
    // Stops short of the light itself, so a surface is never shadowed by the thing lighting it.
    shadow.TMax = distance_to_light - 0.001f;

    ShadowPayload occlusion;
    occlusion.visible = false;
    // Miss shader 1, and no closest hit shader at all: the first hit of any kind ends the
    // search, and only reaching the miss shader means the light is visible.
    TraceRay(scene,
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             0xFFu, 0, 0, 1, shadow, occlusion);

    if (!occlusion.visible) {
        return float3(0.0f);
    }

    // Lambertian, f = albedo / PI. The * light_count undoes the pdf of picking one light
    // uniformly. The * roughness is there for the same reason the bounce direction blends on
    // it: a punctual light has no solid angle, so it contributes nothing to a mirror.
    const float3 brdf = payload.albedo * (1.0f / 3.14159265358979f);
    return throughput * brdf * dot(payload.normal, to_light) * incident *
           float(push.light_count) * payload.roughness;
}

// Nearest area light rectangle along a ray, if any. Area lights are not in the acceleration
// structure -- which is exactly what lets direct_lighting sample them without double counting
// anything -- so seeing the panel at all takes this explicit test.
//
// Only ever called for the camera ray. On a bounce ray the same light has already been
// accounted for by next event estimation at the previous surface, and adding it here as well
// would count it twice.
bool intersect_area_lights(RayDesc ray, out float hit_t, out float3 emission)
{
    hit_t = ray.TMax;
    emission = float3(0.0f);
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
        emission = light.color;
        found = true;
    }

    return found;
}

// ---------------------------------------------------------------------------
// entry points
// ---------------------------------------------------------------------------

// One complete path through the scene, from a jittered point inside `pixel`. The caller
// takes as many of these per frame as push.settings.samples_per_frame asks for; `rng` is
// carried across them so successive samples do not repeat one another.
float3 trace_sample(uint2 pixel, uint2 size, inout Rng rng, out float primary_t,
                    out uint primary_instance)
{
    primary_t = PT_DEPTH_MISS;
    primary_instance = 0u; // 0 means nothing was hit

    // Sub-pixel jitter: free antialiasing, since the frames are being averaged anyway.
    float2 ndc = (float2(pixel) + float2(rand(rng), rand(rng))) / float2(size) * 2.0f - 1.0f;
    ndc.x *= push.aspect;

    const Camera camera = push.camera;

    RayDesc ray;
    ray.Origin = camera.position;
    ray.Direction = normalize(camera.forward +
                              (ndc.x * camera.tan_half_fov) * camera.right -
                              (ndc.y * camera.tan_half_fov) * camera.up);
    ray.TMin = 0.001f;
    ray.TMax = 10000.0f;

    float3 radiance = float3(0.0f);
    float3 throughput = float3(1.0f);

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

        // An area light panel in front of the geometry, seen directly. Camera ray only: on a
        // bounce ray the same light was already accounted for by next event estimation at the
        // previous surface, and adding it again here would count it twice.
        if (bounce == 0u) {
            float light_t;
            float3 light_emission;
            if (intersect_area_lights(ray, light_t, light_emission) &&
                (payload.hit_t < 0.0f || light_t < payload.hit_t)) {
                // The panel is opaque, so this both ends the path and stands in for the hit.
                radiance += throughput * light_emission;
                primary_t = light_t * dot(ray.Direction, camera.forward);
                primary_instance = 0u; // not an entity, so nothing to select
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
        }

        // Unlit: the surface as the material describes it, with no light transport at all.
        // A miss still carries the sky in `emission`, so the background is unchanged.
        if (push.settings.unlit != 0u) {
            radiance = payload.hit_t < 0.0f ? payload.emission
                                            : payload.albedo + payload.emission;
            break;
        }

        radiance += throughput * payload.emission;
        if (payload.hit_t < 0.0f) {
            break;
        }

        const float3 hit_point = ray.Origin + ray.Direction * payload.hit_t;

        // Explicit light sampling, before the albedo is folded into the throughput below --
        // the BRDF inside carries it.
        radiance += direct_lighting(hit_point, payload, throughput, rng);

        throughput *= payload.albedo;

        const float3 diffuse_dir = cosine_hemisphere(payload.normal, rng);
        const float3 mirror_dir = reflect(ray.Direction, payload.normal);

        // Not a physically derived lobe, just a straight blend between the two extremes the
        // scene actually uses (roughness 0 and 1).
        float3 next_dir = normalize(lerp(mirror_dir, diffuse_dir, payload.roughness));
        if (dot(next_dir, payload.normal) <= 0.0f) {
            next_dir = diffuse_dir;
        }

        ray.Origin = hit_point + payload.normal * 0.0005f;
        ray.Direction = next_dir;
        ray.TMin = 0.001f;
        ray.TMax = 10000.0f;

        // Russian roulette, once a path is long enough that the bias-free early exit is
        // worth the added variance.
        if (bounce >= 2u) {
            const float survive = clamp(max(throughput.x, max(throughput.y, throughput.z)),
                                        0.02f, 1.0f);
            if (rand(rng) > survive) {
                break;
            }
            throughput /= survive;
        }
    }

    return radiance;
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
    for (uint sample = 0u; sample < samples; ++sample) {
        float sample_t;
        uint sample_instance;
        radiance += trace_sample(pixel, size, rng, sample_t, sample_instance);
        if (sample == 0u) {
            primary_t = sample_t;
            primary_instance = sample_instance;
        }
    }
    depth[pixel] = primary_t;
    pick[pixel] = primary_instance;

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
    const float3 dir = WorldRayDirection();

    // Vertical gradient plus a broad sun lobe. The lobe is deliberately wide: with no
    // explicit light sampling, a small bright sun would be found only rarely and the image
    // would take far longer to settle.
    const float t = saturate(dir.y * 0.5f + 0.5f);
    const float3 horizon = float3(0.34f, 0.40f, 0.52f);
    const float3 zenith = float3(0.10f, 0.20f, 0.44f);
    float3 sky = lerp(horizon, zenith, t * t);

    const float3 sun_dir = normalize(float3(-0.4f, 0.7f, -0.55f));
    sky += float3(4.0f, 3.6f, 3.0f) * pow(saturate(dot(dir, sun_dir)), 50.0f);

    payload.emission = sky * push.settings.sky_intensity;
    payload.hit_t = -1.0f;
}

// Reached only when a shadow ray got all the way to the light without hitting anything.
[shader("miss")]
void miss_shadow(inout ShadowPayload payload)
{
    payload.visible = true;
}

// Copies the material out of the instance table and orients `normal` against the incoming
// ray, so the integrator never has to care which side it landed on.
void fill_surface(inout Payload payload, float3 normal)
{
    const InstanceData instance = push.instances[InstanceIndex()];

    payload.albedo = instance.albedo;
    payload.emission = instance.emission;
    payload.roughness = instance.roughness;
    payload.hit_t = RayTCurrent();
    payload.normal = dot(normal, WorldRayDirection()) > 0.0f ? -normal : normal;
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
