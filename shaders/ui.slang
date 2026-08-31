// The overlay pass: nuklear's converted vertex stream drawn on top of the presented image.
//
// Compiled by the same rule as the path tracer, so both entry points live in one .spv and
// are selected by name (-fvk-use-entrypoint-name).

// `scale` and `translate` map nuklear's *logical* window pixels to clip space. Feeding
// logical rather than framebuffer units is what makes the overlay keep its physical size on
// a HiDPI display instead of shrinking.
struct PushConstants {
    float2 scale;
    float2 translate;
};
[[vk::push_constant]] PushConstants push;

// Alpha-only font atlas: nuklear bakes NK_FONT_ATLAS_ALPHA8, and the white texel it uses for
// solid shapes lives inside the same image, so one sampler covers text and geometry alike.
[[vk::binding(0, 0)]] uniform Sampler2D atlas;

struct VertexInput {
    [[vk::location(0)]] float2 position;
    [[vk::location(1)]] float2 uv;
    [[vk::location(2)]] float4 color; // R8G8B8A8_UNORM, so already in [0,1]
};

struct Varyings {
    float4 clip : SV_Position;
    [[vk::location(0)]] float2 uv;
    [[vk::location(1)]] float4 color;
};

// The swapchain is an sRGB format, so the colour attachment write encodes linear -> sRGB on
// its own. Nuklear's palette is authored as sRGB bytes, so without undoing that encode here
// every widget would come out visibly washed out.
float3 srgb_to_linear(float3 c)
{
    const float3 low = c / 12.92f;
    const float3 high = pow(max((c + 0.055f) / 1.055f, 0.0f), 2.4f);
    return lerp(low, high, step(0.04045f, c));
}

[shader("vertex")]
Varyings ui_vertex(VertexInput input)
{
    Varyings output;
    output.clip = float4(input.position * push.scale + push.translate, 0.0f, 1.0f);
    output.uv = input.uv;
    // Alpha stays as authored: it is a blend weight, not a colour.
    output.color = float4(srgb_to_linear(input.color.rgb), input.color.a);
    return output;
}

[shader("fragment")]
float4 ui_fragment(Varyings input) : SV_Target
{
    return input.color * float4(1.0f, 1.0f, 1.0f, atlas.Sample(input.uv).r);
}
