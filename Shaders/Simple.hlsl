//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-06 17:46:26
//

#pragma rt_library

struct Camera
{
    column_major float4x4 InvView;
    column_major float4x4 InvProj;

    float3 CameraPosition;
    float Pad;
};

struct PushConstants
{
    int nRenderTarget;
    int nAccel;
    int nCamera;
    int Pad;
};

ConstantBuffer<PushConstants> bConstants : register(b0);

struct RayPayload
{
    float4 vColor;
};

void GenerateCameraRay(uint2 index, out float3 origin, out float3 direction)
{
    ConstantBuffer<Camera> Matrices = ResourceDescriptorHeap[bConstants.nCamera];

    const float2 pixelCenter = DispatchRaysIndex().xy + 0.5;
    const float2 inUV = pixelCenter / DispatchRaysDimensions().xy;
    float2 d = inUV * 2.0 - 1.0;

    float4 orig = mul(Matrices.InvView, float4(0, 0, 0, 1));
    float4 target = mul(Matrices.InvProj, float4(d.x, -d.y, 1, 1));
    float4 dir = mul(Matrices.InvView, float4(normalize(target.xyz), 0));

    origin = orig.xyz;
    direction = dir.xyz;
}

[shader("raygeneration")]
void RayGeneration()
{
    // Resources
    RWTexture2D<float4> tOutput = ResourceDescriptorHeap[bConstants.nRenderTarget];
    RaytracingAccelerationStructure asScene = ResourceDescriptorHeap[bConstants.nAccel];

    uint2 index = DispatchRaysIndex().xy;

    float3 vOrigin = 0.0;
    float3 vDirection = 0.0;
    GenerateCameraRay(index, vOrigin, vDirection);

    RayDesc ray;
    ray.Origin = vOrigin;
    ray.Direction = vDirection;
    ray.TMin = 0.001;
    ray.TMax = 1000.0;

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
