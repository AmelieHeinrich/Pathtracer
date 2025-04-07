//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 01:01:51
//

#include "Model.hpp"

#include <Oslo/Core/Assert.hpp>
#include <Oslo/RHI/Uploader.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

void GLTF::Load(const std::string& path)
{
    Path = path;
    Directory = path.substr(0, path.find_last_of('/'));

    cgltf_options options = {};
    cgltf_data* data = nullptr;

    ASSERT(cgltf_parse_file(&options, path.c_str(), &data) == cgltf_result_success, "Failed to parse GLTF file!");
    ASSERT(cgltf_load_buffers(&options, data, path.c_str()) == cgltf_result_success, "Failed to load GLTF buffers!");
    cgltf_scene *scene = data->scene;

    Root = new GLTFNode;
    Root->Name = "RootNode";
    Root->Parent = nullptr;
    Root->Transform = glm::mat4(1.0f);
    Root->Children.resize(scene->nodes_count);

    for (int i = 0; i < scene->nodes_count; i++) {
        Root->Children[i] = new GLTFNode;
        Root->Children[i]->Parent = Root;
        Root->Children[i]->Transform = glm::mat4(1.0f);

        ProcessNode(scene->nodes[i], Root->Children[i]);
    }
}

GLTF::~GLTF()
{
    FreeNodes(Root);
}

void GLTF::FreeNodes(GLTFNode* node)
{
    if (!node)
        return;

    for (GLTFNode* child : node->Children) {
        FreeNodes(child);
    }
    node->Children.clear();

    delete node;
}


void GLTF::ProcessNode(cgltf_node *node, GLTFNode *mnode)
{
    glm::mat4 localTransform(1.0f);
    glm::mat4 translationMatrix(1.0f);
    glm::mat4 rotationMatrix(1.0f);
    glm::mat4 scaleMatrix(1.0f);

    if (node->has_translation) {
        glm::vec3 translation = glm::vec3(node->translation[0], node->translation[1], node->translation[2]);
        translationMatrix = glm::translate(glm::mat4(1.0f), translation);
    }
    if (node->has_rotation) {
        rotationMatrix = glm::mat4_cast(glm::quat(node->rotation[3], node->rotation[0], node->rotation[1], node->rotation[2]));
    }
    if (node->has_scale) {
        glm::vec3 scale = glm::vec3(node->scale[0], node->scale[1], node->scale[2]);
        scaleMatrix = glm::scale(glm::mat4(1.0f), scale);
    }

    if (node->has_matrix) {
        localTransform *= glm::make_mat4(node->matrix);
    } else {
        localTransform *= translationMatrix * rotationMatrix * scaleMatrix;
    }

    mnode->Name = node->name ? node->name : "Unnamed Node " + std::to_string(rand());
    mnode->Transform = localTransform;

    if (node->mesh) {
        for (int i = 0; i < node->mesh->primitives_count; i++) {
            ProcessPrimitive(&node->mesh->primitives[i], mnode);
        }
    }

    mnode->Children.resize(node->children_count);
    for (int i = 0; i < node->children_count; i++) {
        mnode->Children[i] = new GLTFNode;
        mnode->Children[i]->Parent = mnode;

        ProcessNode(node->children[i], mnode->Children[i]);
    }
}

void GLTF::ProcessPrimitive(cgltf_primitive *primitive, GLTFNode *node)
{
    if (primitive->type != cgltf_primitive_type_triangles) {
        return;
    }

    GLTFPrimitive out;

    cgltf_attribute* posAttribute = nullptr;
    cgltf_attribute* uvAttribute = nullptr;
    cgltf_attribute* normAttribute = nullptr;

    for (int i = 0; i < primitive->attributes_count; i++) {
        if (!strcmp(primitive->attributes[i].name, "POSITION")) {
            posAttribute = &primitive->attributes[i];
        }
        if (!strcmp(primitive->attributes[i].name, "TEXCOORD_0")) {
            uvAttribute = &primitive->attributes[i];
        }
        if (!strcmp(primitive->attributes[i].name, "NORMAL")) {
            normAttribute = &primitive->attributes[i];
        }
    }

    int vertexCount = posAttribute->data->count;
    int indexCount = primitive->indices->count;

    std::vector<Vertex> vertices = {};
    std::vector<uint32_t> indices = {};

    for (int i = 0; i < vertexCount; i++) {
        Vertex vertex = {};

        if (!cgltf_accessor_read_float(posAttribute->data, i, glm::value_ptr(vertex.Position), 4)) {
            vertex.Position = glm::vec3(0.0f);
        }
        if (uvAttribute) {
            if (!cgltf_accessor_read_float(uvAttribute->data, i, glm::value_ptr(vertex.UV), 4)) {
                vertex.UV = glm::vec2(0.0f);
            }
        } else {
            vertex.UV = glm::vec2(0.0f);
        }
        if (!cgltf_accessor_read_float(normAttribute->data, i, glm::value_ptr(vertex.Normal), 4)) {
            vertex.Normal = glm::vec3(0.0f, 0.0f, 1.0f);
        }

        vertices.push_back(vertex);
    }

    for (int i = 0; i < indexCount; i++) {
        indices.push_back(cgltf_accessor_read_index(primitive->indices, i));
    }

    out.VertexCount = vertexCount;
    out.IndexCount = indexCount;

    /// @note(ame): create buffers
    out.VertexBuffer = std::make_shared<Buffer>(vertices.size() * sizeof(Vertex), sizeof(Vertex), BufferType::Storage, node->Name + " Vertex Buffer");
    out.VertexBuffer->BuildSRV();

    out.IndexBuffer = std::make_shared<Buffer>(indices.size() * sizeof(uint32_t), sizeof(uint32_t), BufferType::Storage, node->Name + " Index Buffer");
    out.IndexBuffer->BuildSRV();

    out.GeometryStructure = std::make_shared<BLAS>(out.VertexBuffer, out.IndexBuffer, out.VertexCount, out.IndexCount, node->Name + " BLAS");

    Uploader::EnqueueBufferUpload(vertices.data(), out.VertexBuffer->GetSize(), out.VertexBuffer);
    Uploader::EnqueueBufferUpload(indices.data(), out.IndexBuffer->GetSize(), out.IndexBuffer);
    Uploader::EnqueueAccelerationStructureBuild(out.GeometryStructure);

    out.Instance = {};
    out.Instance.AccelerationStructure = out.GeometryStructure->GetAddress();
    out.Instance.InstanceMask = 1;
    out.Instance.InstanceID = 0;
    out.Instance.Transform = glm::mat3x4(glm::transpose(node->Transform));
    out.Instance.Flags = 0x4;

    VertexCount += out.VertexCount;
    IndexCount += out.IndexCount;

    node->Primitives.push_back(out);
}

void GLTF::TraverseNode(GLTFNode* root, const std::function<void(GLTFNode*)>& fn)
{
    if (!root) {
        return;
    }
    fn(root);

    if (!root->Children.empty()) {
        for (GLTFNode* child : root->Children) {
            TraverseNode(child, fn);
        }
    }
}
