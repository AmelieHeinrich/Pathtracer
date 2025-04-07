//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 14:55:06
//

#pragma once

#include <Oslo/Oslo.hpp>

#include "Model.hpp"

struct Entity
{
    glm::mat4 Transform;
    GLTF Model;
};

class Scene
{
public:
    Scene() = default;
    ~Scene();

    void Build();
    Entity* PushEntity(glm::mat4 transform, const std::string& path);

    std::shared_ptr<TLAS> TopLevelAS;
private:
    std::vector<Entity*> Entities;
    std::vector<RaytracingInstance> Instances;
    std::shared_ptr<Buffer> InstanceBuffer;
};
