//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-06 17:46:26
//

#include "Shaders/Random.hlsl"

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
    int NormalIndex;
    int PBRIndex;
    int Pad;
};

struct Vertex
{
    float3 Position;
    float3 Normal;
    float2 UV;
    float3 Tangent;
    float3 Bitangent;
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
    int nFrameIndex;
    int nSamplesPerPixel;

    int nBouncePerRay;
    int3 Pad;
};

ConstantBuffer<PushConstants> bConstants : register(b0);

struct RayPayload
{
    float3 Throughput;
    float3 AccumulatedColor;

    int Bounce;
    bool Alive;
    float3 NewDirection;
    float3 NewOrigin;

    RNG rng;
};

float3 GetNormalFromNormalMap(int normalIndex, float2 uv, float3 normal, float3 tangent, float3 bitangent)
{
    if (normalIndex == -1)
        return normalize(normal);

    Texture2D<float4> normalMap = ResourceDescriptorHeap[normalIndex];
    SamplerState sampler = SamplerDescriptorHeap[bConstants.nWrapSampler];

    // Sample the normal map (assumes normals are stored in [0,1] range)
    float3 normalSample = normalMap.SampleLevel(sampler, uv, 0.0).rgb;

    // Transform from [0,1] to [-1,1]
    normalSample = normalSample * 2.0f - 1.0f;

    // Construct the TBN matrix
    float3x3 TBN = float3x3(tangent, bitangent, normal);

    // Transform the normal from tangent space to world space
    float3 worldNormal = normalize(mul(normalSample, TBN));

    return worldNormal;
}

float2 GetMetallicRoughness(int pbrIndex, float2 uv)
{
    if (pbrIndex == -1)
        return float2(0, 0.5);

    Texture2D<float4> pbrMap = ResourceDescriptorHeap[pbrIndex];
    SamplerState sampler = SamplerDescriptorHeap[bConstants.nWrapSampler];

    float4 data = pbrMap.SampleLevel(sampler, uv, 0.0);
    return float2(data.b, data.g);
}

[shader("raygeneration")]
void RayGeneration()
{
    // Resources
    ConstantBuffer<Camera> Matrices = ResourceDescriptorHeap[bConstants.nCamera];
    RWTexture2D<float4> tOutput = ResourceDescriptorHeap[bConstants.nRenderTarget];
    RaytracingAccelerationStructure asScene = ResourceDescriptorHeap[bConstants.nAccel];

    uint2 index = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;

    float4 pixelColor = 0;

    uint samplesPerPixel = bConstants.nSamplesPerPixel;
    for (uint sample = 0; sample < samplesPerPixel; sample++) {
        // Generate RNG and payload
        RayPayload payload = (RayPayload)0;
        payload.rng = rng_init(index, bConstants.nFrameIndex * 7919 + sample * 104729); // better variation
        payload.Alive = true;
        payload.Throughput = 1.0;
        payload.AccumulatedColor = 0;

        // Generate ray
        float2 offset = float2(next_float(payload.rng) - 0.5, next_float(payload.rng) - 0.5);

        float3 vOrigin = 0.0;
        float3 vDirection = 0.0;

        const float2 pixelCenter = (index.xy + 0.5) + offset;
        const float2 inUV = pixelCenter / dimensions.xy;
        float2 d = (inUV * 2.0 - 1.0);

        vOrigin = mul(Matrices.InvView, float4(0, 0, 0, 1)).xyz;
        float4 target = mul(Matrices.InvProj, float4(d.x, -d.y, 1, 1));
        vDirection = mul(Matrices.InvView, float4(normalize(target.xyz), 0)).xyz;

        RayDesc ray;
        ray.TMin = 0.001;
        ray.TMax = 1000.0;

        ray.Origin = vOrigin;
        ray.Direction = vDirection;

        payload.NewOrigin = vOrigin;
        payload.NewDirection = vDirection;

        // Trace bounces
        for (int bounce = 0; bounce < bConstants.nBouncePerRay; bounce++) {
            if (!payload.Alive) {
                break;
            }

            payload.Bounce = bounce;
            TraceRay(
                asScene,
                RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                0xFF,
                0,
                0,
                0,
                ray,
                payload
            );

            ray.Origin = payload.NewOrigin;
            ray.Direction = payload.NewDirection;
        }

        pixelColor += float4(payload.AccumulatedColor, 1.0);
    }

    tOutput[index] = pixelColor / samplesPerPixel;
}

[shader("closesthit")]
void ClosestHit(inout RayPayload Payload, in BuiltInTriangleIntersectionAttributes Attr)
{
    float3 hitPos = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();

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

    // Attributes
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
    float3 tangent = normalize(
        (1.0 - Attr.barycentrics.x - Attr.barycentrics.y) * v0.Tangent +
        Attr.barycentrics.x * v1.Tangent +
        Attr.barycentrics.y * v2.Tangent
    );
    float3 bitangent = normalize(
        (1.0 - Attr.barycentrics.x - Attr.barycentrics.y) * v0.Bitangent +
        Attr.barycentrics.x * v1.Bitangent +
        Attr.barycentrics.y * v2.Bitangent
    );
    normal = GetNormalFromNormalMap(material.NormalIndex, uv, normal, tangent, bitangent);

    // Set new dir
    float3 direction = next_unit_on_hemisphere(Payload.rng, normal);
    Payload.NewDirection = direction;
    Payload.NewOrigin = hitPos + (normal * 0.001);
    
    // Shade
    float3 albedo = tAlbedo.SampleLevel(sSampler, uv, 0.0).rgb;
    float cosTheta = dot(normal, direction);
    float pdf = cosTheta / 3.14159;
    float3 f_r = albedo / 3.14159;
    
    Payload.Throughput *= f_r * cosTheta / pdf;
    Payload.AccumulatedColor += Payload.Throughput * 0; // No radiance
}

[shader("miss")]
void Miss(inout RayPayload Payload)
{
    SamplerState sCubeSampler = SamplerDescriptorHeap[bConstants.nWrapSampler];
    TextureCube<float4> tEnvironment = ResourceDescriptorHeap[bConstants.nCubemap];

    Payload.AccumulatedColor += Payload.Throughput * tEnvironment.SampleLevel(sCubeSampler, normalize(WorldRayDirection()), 0).rgb;
    Payload.Alive = false;
}

[shader("anyhit")]
void AnyHit(inout RayPayload Payload, in BuiltInTriangleIntersectionAttributes Attr)
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

    float4 albedo = tAlbedo.SampleLevel(sSampler, uv, 0.0);
    if (albedo.a < 0.5)
        IgnoreHit();
}
