//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-08 19:11:38
//

// Credit to Dihara Wijetunga
// https://github.com/diharaw/hybrid-rendering/blob/master/src/shaders/random.glsl

struct RNG
{
    uint2 s;
};

// xoroshiro64* random number generator.
// http://prng.di.unimi.it/xoroshiro64star.c
uint rng_rotl(uint x, uint k)
{
    return (x << k) | (x >> (32 - k));
}

// Xoroshiro64* RNG
uint rng_next(inout RNG rng)
{
    uint result = rng.s.x * 0x9e3779bb;

    rng.s.y ^= rng.s.x;
    rng.s.x = rng_rotl(rng.s.x, 26) ^ rng.s.y ^ (rng.s.y << 9);
    rng.s.y = rng_rotl(rng.s.y, 13);

    return result;
}

// Thomas Wang 32-bit hash.
// http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
uint rng_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

RNG rng_init(uint2 id, uint frameIndex)
{
    uint s0 = (id.x << 16) | id.y;
    uint s1 = frameIndex;

    RNG rng;
    rng.s.x = rng_hash(s0);
    rng.s.y = rng_hash(s1);
    rng_next(rng);
    return rng;
}

float next_float(inout RNG rng)
{
    uint u = 0x3f800000 | (rng_next(rng) >> 9);
    return asfloat(u) - 1.0;
}

uint next_uint(inout RNG rng, uint nmax)
{
    float f = next_float(rng);
    return uint(floor(f * nmax));
}

float2 next_vec2(inout RNG rng)
{
    return float2(next_float(rng), next_float(rng));
}

float3 next_vec3(inout RNG rng)
{
    return float3(next_float(rng), next_float(rng), next_float(rng));
}

float3 next_unit_vector(inout RNG rng)
{
    float z = 1.0 - 2.0 * next_float(rng);
    float phi = 2.0 * 3.14159 * next_float(rng);
    float r = sqrt(1.0 - z * z);

    return float3(r * cos(phi), r * sin(phi), z);
}

float3 next_unit_on_hemisphere(inout RNG rng, float3 normal)
{
    float3 onUnitSphere = next_unit_vector(rng);
    if (dot(onUnitSphere, normal) > 0.0)
        return onUnitSphere;
    return -onUnitSphere;
}
