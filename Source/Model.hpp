//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 00:59:59
//

#pragma once

#include <Oslo/Oslo.hpp>
#include <Oslo/RHI/BLAS.hpp>

#include <cgltf.h>
#include <glm/glm.hpp>
#include <functional>

#include "Util/TangentCalculator.hpp"

struct RaytracingMaterial
{
    int AlbedoIndex;
    glm::ivec3 Pad;
};

struct GLTFMaterial
{
    std::shared_ptr<Texture> Albedo;
    std::shared_ptr<View> AlbedoView;

    bool AlphaTested;
};

struct GLTFPrimitive
{
    std::shared_ptr<Buffer> VertexBuffer;
    std::shared_ptr<Buffer> IndexBuffer;

    RaytracingInstance Instance;
    std::shared_ptr<BLAS> GeometryStructure;

    uint32_t VertexCount;
    uint32_t IndexCount;
    int MaterialIndex;
};

struct GLTFNode
{
    std::vector<GLTFPrimitive> Primitives;

    std::string Name = "";
    glm::mat4 Transform;
    GLTFNode* Parent = nullptr;
    std::vector<GLTFNode*> Children = {};

    GLTFNode() = default;
};

class GLTF
{
public:
    std::string Path;
    std::string Directory;

    GLTFNode* Root = nullptr;
    std::vector<GLTFMaterial> Materials;
    std::shared_ptr<Buffer> MaterialBuffer;

    uint32_t VertexCount = 0;
    uint32_t IndexCount = 0;

    void Load(const std::string& path);
    ~GLTF();

    void TraverseNode(GLTFNode* root, const std::function<void(GLTFNode*)>& fn);
private:
    void ProcessPrimitive(cgltf_primitive *primitive, GLTFNode *node);
    void ProcessNode(cgltf_node *node, GLTFNode *mnode);
    void FreeNodes(GLTFNode* node);
};
