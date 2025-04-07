//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 19:57:36
//

#include "GlobalResources.hpp"

void GlobalResources::PushModel(GLTF& gltf)
{
    gltf.TraverseNode(gltf.Root, [&](GLTFNode* root) {
        for (auto& primitive : root->Primitives) {
            primitive.Instance.InstanceID = mInstanceCount;
            
            Instance instance;
            instance.VertexBuffer = primitive.VertexBuffer->SRV();
            instance.IndexBuffer = primitive.IndexBuffer->SRV();
            instance.MaterialIndex = 0;
            instance.Pad = 0;

            mInstances.push_back(instance);
            
            mInstanceCount++;
        }
    });
}

void GlobalResources::Build()
{
    InstanceBuffer = std::make_shared<Buffer>(sizeof(Instance) * mInstances.size(), sizeof(Instance), BufferType::Storage, "Instance Buffer");
    InstanceBuffer->BuildSRV();
    // MaterialBuffer = std::make_shared<Buffer>(sizeof(Material) * mMaterials.size(), sizeof(Material), BufferType::Storage, "Material Buffer");

    Uploader::EnqueueBufferUpload(mInstances.data(), mInstances.size() * sizeof(Instance), InstanceBuffer);
    // Uploader::EnqueueBufferUpload(mMaterials.data(), mMaterials.size() * sizeof(Material), MaterialBuffer);
}
