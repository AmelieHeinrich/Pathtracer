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
    int Pad;
};

struct Material
{
    int AlbedoIndex;
    int NormalIndex;
    int PBRIndex;
    int Pad;
};

class GlobalResources
{
public:
    std::shared_ptr<Buffer> InstanceBuffer;
    std::shared_ptr<Buffer> MaterialBuffer;

    void PushModel(GLTF& gltf);
    void Build();
private:
    std::vector<Instance> mInstances;
    std::vector<Material> mMaterials;
    
    int mInstanceCount = 0;
    int mMaterialCount = 0;
};
