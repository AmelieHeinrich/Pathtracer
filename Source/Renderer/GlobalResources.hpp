//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 19:51:52
//

#pragma once

#include <Oslo/Oslo.hpp>

#include "Model.hpp"

/*
    With the help of bindless resources, we can store materials and instance data into one huge ass array that we can then use in our raytracing shader.
*/

struct Instance
{
    int VertexBuffer;
    int IndexBuffer;
    int MaterialIndex;
    int MaterialBuffer;
};

class GlobalResources
{
public:
    std::shared_ptr<Buffer> InstanceBuffer;

    void PushModel(GLTF& gltf);
    void Build();
private:
    std::vector<Instance> mInstances;
    
    int mInstanceCount = 0;
};
