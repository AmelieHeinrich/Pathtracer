//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-06 17:46:26
//

#pragma rt_library

struct PushConstants
{
    int nRenderTarget;
    int3 nPad;
};

ConstantBuffer<PushConstants> bConstants : register(b0);
RaytracingAccelerationStructure asScene : register(b1);

struct [raypayload] RayPayload
{
    float4 vColor : read(caller) : write(caller);
};

[shader("raygeneration")]
void RayGeneration()
{
    // Resources
    RWTexture2D<float4> tOutput = ResourceDescriptorHeap[bConstants.nRenderTarget];

    uint2 dispatchIndex = DispatchRaysIndex().xy;

    // Orthographic projection since we're raytracing in screen space.
    float3 vRayDir = float3(0, 0, 1);
    float3 vOrigin = float3(dispatchIndex.x, dispatchIndex.y, 0.0f);

    RayDesc ray;
    ray.Origin = vOrigin;
    ray.Direction = vRayDir;
    ray.TMin = 0.01;
    ray.TMax = 10000.0;

    RayPayload Payload = { float4(0, 0, 0, 1) };
    TraceRay(
        asScene,
        RAY_FLAG_NONE,
        0xFF,
        0,
        0,
        0,
        ray,
        Payload
    );

    // Write the raytraced color to the output texture.
    tOutput[dispatchIndex] = Payload.vColor;
}

[shader("closesthit")]
void ClosestHit(inout RayPayload Payload, in BuiltInTriangleIntersectionAttributes Attr)
{
    float3 vBarycentrics = float3(1 - Attr.barycentrics.x - Attr.barycentrics.y, Attr.barycentrics.x, Attr.barycentrics.y);
    Payload.vColor = float4(vBarycentrics, 1);
}

[shader("miss")]
void Miss(inout RayPayload Payload)
{
    float slope = normalize(WorldRayDirection()).y;
    float t = saturate(slope * 5 + 0.5);
    Payload.vColor = float4(lerp(float3(0.75, 0.86, 0.93), float3(0.24, 0.44, 0.72), t), 1.0);
}
