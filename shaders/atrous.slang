// Edge-avoiding a-trous wavelet filter, after Dammertz et al. 2010, "Edge-Avoiding A-Trous
// Wavelet Transform for fast Global Illumination Filtering".
//
// One pass of a 5x5 B3-spline blur whose taps are spread `step` pixels apart, weighted down
// wherever the neighbour disagrees with the centre in luminance, normal or depth. The caller
// runs it several times with step doubling each time, which reaches a wide support -- 65
// pixels after five iterations -- for the cost of 25 taps a pass rather than 65x65.
//
// This filters the tracer's already-averaged output image. It is a view filter: nothing here
// feeds back into the accumulator, so switching it on costs no converged samples.

struct PushConstants {
    uint width;
    uint height;
    // Tap spacing in pixels: 1, 2, 4, ... The "a trous" -- with holes -- of the name.
    uint step;
    // Already divided by 2^iteration by the caller, per the paper: the colour weight has to
    // tighten as the taps spread, or the last and widest passes smear real detail away.
    float phi_color;
    float phi_normal;
    float phi_depth;
};
[[vk::push_constant]] PushConstants push;

[[vk::binding(0, 0)]] uniform RWTexture2D<float4> src;
[[vk::binding(1, 0)]] uniform RWTexture2D<float4> dst;
// Both are the tracer's primary-hit buffers, read only. A miss leaves PT_DEPTH_MISS in depth
// and the negated ray direction in the normal, which is what lets sky filter against sky
// while never filtering against geometry.
[[vk::binding(2, 0)]] uniform RWTexture2D<float4> normal_buffer;
[[vk::binding(3, 0)]] uniform RWTexture2D<float> depth_buffer;

// The separable B3-spline kernel {1, 4, 6, 4, 1} / 16, applied as an outer product below.
static const float KERNEL[5] = {1.0f / 16.0f, 1.0f / 4.0f, 3.0f / 8.0f, 1.0f / 4.0f,
                                1.0f / 16.0f};

// Rec. 709 luminance. The colour weight compares brightness rather than the full RGB triple
// because that is where Monte Carlo noise lives; comparing chroma as well would let a noisy
// pixel defend itself from being filtered.
float luminance(float3 color)
{
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void atrous(uint3 thread : SV_DispatchThreadID)
{
    const uint2 pixel = thread.xy;
    // The dispatch is rounded up to whole 8x8 groups, so the tail threads have no pixel.
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    const float3 center_color = src[pixel].rgb;
    const float3 center_normal = normal_buffer[pixel].xyz;
    const float center_depth = depth_buffer[pixel];
    const float center_luma = luminance(center_color);

    // The depth tolerance widens with the tap spacing: neighbours further apart are expected
    // to differ more on a surface seen at an angle, and a fixed tolerance would reject the
    // whole of a slanted floor by the last iteration.
    const float depth_scale = push.phi_depth * float(push.step) + 1e-4f;

    float3 sum = float3(0.0f);
    float sum_weight = 0.0f;

    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            // Clamping rather than skipping keeps the kernel normalised at the borders --
            // sum_weight divides it out below either way, but clamping avoids darkening the
            // edge of the image where half the taps would otherwise be missing.
            const int2 offset = int2(dx, dy) * int(push.step);
            const int2 tap = clamp(int2(pixel) + offset, int2(0, 0),
                                   int2(int(push.width) - 1, int(push.height) - 1));
            const uint2 coord = uint2(tap);

            const float3 tap_color = src[coord].rgb;
            const float3 tap_normal = normal_buffer[coord].xyz;
            const float tap_depth = depth_buffer[coord];

            // Squared luminance difference, as in the paper. A firefly is a large difference
            // and so is mostly rejected -- which is the honest behaviour: this filter removes
            // noise the estimator has already averaged down, it does not clamp outliers.
            const float luma_delta = luminance(tap_color) - center_luma;
            const float w_color = exp(-(luma_delta * luma_delta) / push.phi_color);

            // A cosine raised to a large power: near 1 while the surfaces face the same way,
            // and falling off hard across a crease.
            const float w_normal =
                pow(max(0.0f, dot(center_normal, tap_normal)), push.phi_normal);

            // PT_DEPTH_MISS is 1e30, a finite value, so sky against sky gives exactly zero
            // here and sky against geometry gives an exponent no float survives -- exp
            // underflows to 0 rather than producing a NaN.
            const float w_depth = exp(-abs(tap_depth - center_depth) / depth_scale);

            const float weight = KERNEL[dx + 2] * KERNEL[dy + 2] * w_color * w_normal * w_depth;
            sum += weight * tap_color;
            sum_weight += weight;
        }
    }

    // The centre tap weighs itself: its three edge terms are all exactly 1, so sum_weight is
    // at least 3/8 * 3/8 and the division is never anywhere near zero.
    dst[pixel] = float4(sum / max(sum_weight, 1e-8f), 1.0f);
}
