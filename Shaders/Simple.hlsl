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

struct RayPayload
{
    float4 vColor;
};

[shader("raygeneration")]
void RayGeneration()
{
    // Resources
    RWTexture2D<float4> tOutput = ResourceDescriptorHeap[bConstants.nRenderTarget];

    float2 lerpValues = (float2)DispatchRaysIndex() / (float2)DispatchRaysDimensions();

    // Orthographic projection since we're raytracing in screen space.
    float3 vRayDir = float3(0, 0, 1);
    float3 vOrigin = float3(
        lerp(-1.0f, 1.0f, lerpValues.x),
        lerp(-1.0f, 1.0f, lerpValues.y),
        0.0f
    );

    RayDesc ray;
    ray.Origin = vOrigin;
    ray.Direction = vRayDir;
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    RayPayload Payload = { float4(0, 0, 0, 1) };
    TraceRay(
        asScene,
        RAY_FLAG_NONE,
        ~0,
        0,
        0,
        0,
        ray,
        Payload
    );

    // Write the raytraced color to the output texture.
    tOutput[DispatchRaysIndex().xy] = Payload.vColor;
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
    Payload.vColor = float4(1, 0, 1, 1);
}
