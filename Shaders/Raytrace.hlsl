//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-06 17:46:26
//

#pragma rt_library

struct Instance
{
    int VertexBuffer;
    int IndexBuffer;
    int MaterialIndex;
    int MaterialBuffer;
};

struct Material
{
    int AlbedoIndex;
    int3 Pad;
};

struct Vertex
{
    float3 Position;
    float3 Normal;
    float2 UV;
};

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
    int nInstanceBuffer;

    int nWrapSampler;
    int nCubemap;
    int2 Pad;
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
        RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
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
    StructuredBuffer<Instance> bInstances = ResourceDescriptorHeap[bConstants.nInstanceBuffer];
    Instance instance = bInstances[InstanceIndex()];
    
    StructuredBuffer<Material> bMaterials = ResourceDescriptorHeap[instance.MaterialBuffer];
    Material material = bMaterials[instance.MaterialIndex];
    
    Texture2D<float4> tAlbedo = ResourceDescriptorHeap[material.AlbedoIndex];
    SamplerState sSampler = SamplerDescriptorHeap[bConstants.nWrapSampler];

    StructuredBuffer<Vertex> bVertices = ResourceDescriptorHeap[instance.VertexBuffer];
    StructuredBuffer<uint> bIndices = ResourceDescriptorHeap[instance.IndexBuffer];

    uint3 indices = uint3(
        bIndices[PrimitiveIndex() * 3 + 0],
        bIndices[PrimitiveIndex() * 3 + 1],
        bIndices[PrimitiveIndex() * 3 + 2]
    );

    Vertex v0 = bVertices[indices.x];
    Vertex v1 = bVertices[indices.y];
    Vertex v2 = bVertices[indices.z];

    float3 bary = float3(
        1.0 - Attr.barycentrics.x - Attr.barycentrics.y,
        Attr.barycentrics.x,
        Attr.barycentrics.y
    );
    float2 uv = v0.UV * bary.x + v1.UV * bary.y + v2.UV * bary.z;
    float3 normal = normalize(
        (1.0 - Attr.barycentrics.x - Attr.barycentrics.y) * v0.Normal +
        Attr.barycentrics.x * v1.Normal +
        Attr.barycentrics.y * v2.Normal
    );

    Payload.vColor = tAlbedo.SampleLevel(sSampler, uv, 0.0);
}

[shader("miss")]
void Miss(inout RayPayload Payload)
{
    SamplerState sCubeSampler = SamplerDescriptorHeap[bConstants.nWrapSampler];
    TextureCube<float4> tEnvironment = ResourceDescriptorHeap[bConstants.nCubemap];

    Payload.vColor = tEnvironment.SampleLevel(sCubeSampler, normalize(WorldRayDirection()), 0);
}
