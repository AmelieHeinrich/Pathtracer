// Debug lines -- gizmos, wireframes, the ground grid -- drawn over the traced image.
//
// Occlusion does not go through a depth buffer, because there is none: the path tracer writes
// only colour. Instead raygen leaves each pixel's primary hit depth in a storage image, and
// the fragment shader below compares its own against it. So this is a depth test against the
// tracer's own geometry, in world units, needing no projection to be inverted and no depth
// attachment to exist.

struct PushConstants {
    // The view projection as four rows, so `clip.x = dot(rows[0], p)`. Passing rows rather
    // than a float4x4 sidesteps the whole question of matrix layout between C and SPIR-V.
    float4 rows[4];
    // 0 for lines that must stay visible through the scene -- gizmo handles, light markers,
    // the selection wireframe, all of which exist to be seen and grabbed. 1 for lines that
    // describe the world and should sit in it, like the ground grid.
    uint depth_test;
};
[[vk::push_constant]] PushConstants push;

// The same image raygen writes, bound read-only here. A plain Texture2D rather than an
// RWTexture2D because a fragment shader may only carry a *writable* storage image when
// fragmentStoresAndAtomics is enabled, and this shader never writes. Load takes integer pixel
// coordinates directly, so it needs no sampler either.
[[vk::binding(0, 0)]] uniform Texture2D<float> scene_depth;

struct VertexInput {
    [[vk::location(0)]] float3 position;
    [[vk::location(1)]] float4 color; // R8G8B8A8_UNORM
};

struct Varyings {
    float4 clip : SV_Position;
    [[vk::location(0)]] float4 color;
    // Camera space depth, the same quantity raygen stores. Interpolated perspective-correctly,
    // which is an approximation along a line but an entirely adequate one at one pixel wide.
    [[vk::location(1)]] float view_depth;
};

// Matches the conversion the UI shader does, and for the same reason: the swapchain is an
// sRGB format and re-encodes on write, so a colour authored as sRGB bytes has to be undone
// here or it comes out washed out.
float3 srgb_to_linear(float3 c)
{
    const float3 low = c / 12.92f;
    const float3 high = pow(max((c + 0.055f) / 1.055f, 0.0f), 2.4f);
    return lerp(low, high, step(0.04045f, c));
}

[shader("vertex")]
Varyings debug_vertex(VertexInput input)
{
    const float4 world = float4(input.position, 1.0f);

    Varyings output;
    output.clip = float4(dot(push.rows[0], world), dot(push.rows[1], world),
                         dot(push.rows[2], world), dot(push.rows[3], world));
    output.color = float4(srgb_to_linear(input.color.rgb), input.color.a);
    // Row 3 of the view projection is the camera forward axis with the eye folded in, so w is
    // already dot(forward, p - eye) -- exactly the depth raygen stores.
    output.view_depth = output.clip.w;
    return output;
}

[shader("fragment")]
float4 debug_fragment(Varyings input) : SV_Target
{
    if (push.depth_test != 0u) {
        // The third component is the mip level, of which there is only ever one.
        const float scene = scene_depth.Load(int3(int2(input.clip.xy), 0));
        if (input.view_depth > scene) {
            discard;
        }
    }
    return input.color;
}
