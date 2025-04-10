//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-10 22:54:06
//

#pragma kernel CSMain

struct PushConstants
{
    int nHDR;
    int nLDR;
    float fGamma;
    int nPad;
};

ConstantBuffer<PushConstants> Constants : register(b0);

float3 ACESFilm(float3 X)
{
    float A = 2.51f;
    float B = 0.03f;
    float C = 2.43f;
    float D = 0.59f;
    float E = 0.14f;
    return saturate((X * (A * X + B)) / (X * (C * X + D) + E));
}

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    Texture2D<float4> hdr = ResourceDescriptorHeap[Constants.nHDR];
    RWTexture2D<float4> ldr = ResourceDescriptorHeap[Constants.nLDR];

    uint width, height;
    hdr.GetDimensions(width, height);

    if (tid.x <= width && tid.y <= height) {
        float4 hdrColor = hdr.Load(uint3(tid.xy, 0));
        float3 mappedColor = ACESFilm(hdrColor.rgb);
        mappedColor.rgb = pow(mappedColor.rgb, 1.0 / Constants.fGamma);

        ldr[tid.xy] = float4(mappedColor, 1.0);
    }
}

