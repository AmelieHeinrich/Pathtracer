//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-08 19:11:38
//

float random(float2 value)
{
    //make value smaller to avoid artefacts
    float2 smallValue = sin(value);
    //get scalar value from 3d vector
    float random = dot(smallValue, float2(12.9898, 78.233));
    //make value more random by making it bigger and then taking teh factional part
    random = frac(sin(random) * 143758.5453);
    return random;
}
