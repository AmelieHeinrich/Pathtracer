//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 14:55:06
//

#pragma once

#include <Oslo/Oslo.hpp>

#include "Model.hpp"

struct CameraInfo
{
    glm::mat4 View;
    glm::mat4 Projection;
    glm::vec3 Position;
};

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
    CameraInfo CamInfo;
private:
    std::vector<Entity*> Entities;
    std::vector<RaytracingInstance> Instances;
    std::shared_ptr<Buffer> InstanceBuffer;
};
