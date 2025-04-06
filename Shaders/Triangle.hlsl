//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-06 15:39:44
//

#pragma vertex VSMain
#pragma pixel PSMain

static const float3 VERTICES[] = {
    float3(-0.5, -0.5, 0.0),
    float3( 0.5, -0.5, 0.0),
    float3( 0.0,  0.5, 0.0)
};

static const float3 COLORS[] = {
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0)
};

struct VertexOut
{
    float4 vPosition : SV_POSITION;
    float3 vColor : COLOR;
};

VertexOut VSMain(uint nVertexID : SV_VertexID)
{
    VertexOut Out = (VertexOut)0;

    Out.vPosition = float4(VERTICES[nVertexID], 1.0);
    Out.vColor = COLORS[nVertexID];

    return Out;
}

float4 PSMain(VertexOut Out) : SV_Target
{
    return float4(Out.vColor, 1.0);
}
