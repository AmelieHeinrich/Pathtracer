// Scene-referred radiance to display-referred linear, in one compute pass.
//
// The result stays linear on purpose. The swapchain is an sRGB format and the blit into it
// performs the linear-to-sRGB encode in hardware, so a transfer function here would be
// applied twice -- and renderer_screenshot's encode_srgb mirrors that same hardware encode
// for the PNG path. What this shader owns is the *shape* of the response, nothing else.

// Matches tonemap_curve_t in src/tonemap.h.
static const uint CURVE_NONE = 0u;
static const uint CURVE_AGX = 1u;
static const uint CURVE_ACES = 2u;
static const uint CURVE_REINHARD = 3u;

struct PushConstants {
    uint width;
    uint height;
    uint curve;
    // Already exp2'd on the CPU: this is a plain multiplier, not a stop count.
    float exposure_scale;
};
[[vk::push_constant]] PushConstants push;

[[vk::binding(0, 0)]] uniform RWTexture2D<float4> src;
[[vk::binding(1, 0)]] uniform RWTexture2D<float4> dst;

// ---------------------------------------------------------------------------
// AgX
// ---------------------------------------------------------------------------

// AgX works in its own slightly desaturated primaries. Rotating into them before the curve
// is the whole trick: a channel that would have clipped on its own instead drags the other
// two with it, so an overexposed highlight desaturates toward white the way film does rather
// than turning into a flat primary.
static const float3x3 AGX_TO = float3x3(0.842479062253094f, 0.0784335999999992f, 0.0792237451477643f,
                                        0.0423282422610123f, 0.878468636469772f, 0.0791661274605434f,
                                        0.0423756549057051f, 0.0784336f, 0.879142973793104f);

static const float3x3 AGX_FROM = float3x3(1.19687900512017f, -0.0980208811401368f, -0.0990297440797205f,
                                          -0.0528968517574562f, 1.15190312990417f, -0.0989611768448433f,
                                          -0.0529716355144438f, -0.0980434501171241f, 1.15107367264116f);

// The dynamic range AgX maps across, in stops either side of mid grey.
static const float AGX_MIN_EV = -12.47393f;
static const float AGX_MAX_EV = 4.026069f;

// A degree-six polynomial fit to AgX's sigmoid. The real curve is a pair of splines; this is
// the standard approximation and is well within a quantisation step of it over [0,1].
float3 agx_sigmoid(float3 x)
{
    const float3 x2 = x * x;
    const float3 x4 = x2 * x2;
    return 15.5f * x4 * x2 - 40.14f * x4 * x + 31.96f * x4 - 6.868f * x2 * x + 0.4298f * x2 +
           0.1191f * x - 0.00232f;
}

float3 tonemap_agx(float3 color)
{
    float3 v = mul(AGX_TO, color);

    // Log encode. The clamp is what bounds the sigmoid's domain, and the max() keeps log2
    // away from zero -- a negative or zero channel is possible here, both from the spectral
    // upsampling and from the matrix above.
    v = clamp(log2(max(v, 1e-10f)), AGX_MIN_EV, AGX_MAX_EV);
    v = (v - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV);

    v = saturate(agx_sigmoid(v));

    // Back to the rendering primaries. The sigmoid's output is display *encoded*, so undoing
    // that encode is what leaves this pass linear for the blit to re-encode.
    v = mul(AGX_FROM, v);
    return max(pow(max(v, 0.0f), 2.2f), 0.0f);
}

// ---------------------------------------------------------------------------
// the rest
// ---------------------------------------------------------------------------

// Narkowicz's fit to the ACES RRT and ODT. Returns linear, so it pairs with the blit's encode
// exactly the way AgX above does.
float3 tonemap_aces(float3 color)
{
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

float3 tonemap_reinhard(float3 color)
{
    return color / (1.0f + max(color, 0.0f));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void tonemap(uint3 thread : SV_DispatchThreadID)
{
    const uint2 pixel = thread.xy;
    // The dispatch is rounded up to whole 8x8 groups, so the tail threads have no pixel.
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    // NaN fails every comparison, so testing for the inside of the range is what keeps one
    // out of the curves below -- the same reasoning encode_srgb uses on the CPU.
    float3 color = src[pixel].rgb;
    color = color > 0.0f ? color : float3(0.0f);
    color *= push.exposure_scale;

    switch (push.curve) {
    case CURVE_AGX:
        color = tonemap_agx(color);
        break;
    case CURVE_ACES:
        color = tonemap_aces(color);
        break;
    case CURVE_REINHARD:
        color = tonemap_reinhard(color);
        break;
    default:
        // CURVE_NONE: the naked clip the renderer has always applied at the very end.
        color = saturate(color);
        break;
    }

    dst[pixel] = float4(color, 1.0f);
}
