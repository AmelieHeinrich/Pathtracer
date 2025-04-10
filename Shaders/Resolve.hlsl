//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-10 22:49:09
//

#pragma kernel CSMain

struct PushConstants
{
    int nCurrent;
    int nPrevious;
    int nFrameCount;
    int nPad;
};

ConstantBuffer<PushConstants> Constants : register(b0);

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    Texture2D<float4> old = ResourceDescriptorHeap[Constants.nPrevious];
    RWTexture2D<float4> current = ResourceDescriptorHeap[Constants.nCurrent];

    uint width, height;
    current.GetDimensions(width, height);

    if (tid.x <= width && tid.y <= height) {
        current[tid.xy] = lerp(old.Load(uint3(tid.xy, 0)), current[tid.xy], 1.0 / Constants.nFrameCount);
    }
}
