//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 15:11:27
//

#include "Scene.hpp"

Scene::~Scene()
{
    for (auto& entity : Entities) {
        delete entity;
    }
    Entities.clear();
}

void Scene::Build()
{
    Resources.Build();

    for (auto& entity : Entities) {
        entity->Model.TraverseNode(entity->Model.Root, [&](GLTFNode* node){
            for (auto& primitive : node->Primitives) {
                GLTFMaterial material = entity->Model.Materials[primitive.MaterialIndex];

                glm::mat4 transform = glm::mat4(primitive.Instance.Transform) * entity->Transform;
                primitive.Instance.Transform = glm::mat3x4(transform);
                primitive.Instance.Flags = material.AlphaTested ? D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE : D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE;
                Instances.push_back(primitive.Instance);
            }
        });
    }

    InstanceBuffer = std::make_shared<Buffer>(sizeof(RaytracingInstance) * Instances.size(), sizeof(RaytracingInstance), BufferType::Constant, "Scene Instances");
    InstanceBuffer->CopyMapped(Instances.data(), sizeof(RaytracingInstance) * Instances.size());

    TopLevelAS = std::make_shared<TLAS>(InstanceBuffer, Instances.size(), "Scene TLAS");
    Uploader::EnqueueAccelerationStructureBuild(TopLevelAS);
}

Entity* Scene::PushEntity(glm::mat4 transform, const std::string& path)
{
    Entity* entity = new Entity;
    entity->Model.Load(path);
    entity->Transform = transform;
    Entities.push_back(entity);

    Resources.PushModel(entity->Model);

    return entity;
}
