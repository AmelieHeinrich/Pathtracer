//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 01:01:51
//

#include "Model.hpp"
#include "Cache/TextureCache.hpp"
#include "Renderer/RendererTools.hpp"

#include <Oslo/Core/Assert.hpp>
#include <Oslo/RHI/Uploader.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

void ComputeTangentSpace(std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices)
{
    std::vector<glm::vec3> accumulatedTangents(vertices.size(), {0, 0, 0});
    std::vector<glm::vec3> accumulatedBitangents(vertices.size(), {0, 0, 0});

    for (size_t i = 0; i < indices.size(); i += 3) {
        uint32_t i0 = indices[i];
        uint32_t i1 = indices[i + 1];
        uint32_t i2 = indices[i + 2];

        const Vertex& v0 = vertices[i0];
        const Vertex& v1 = vertices[i1];
        const Vertex& v2 = vertices[i2];

        glm::vec3 p0 = v0.Position;
        glm::vec3 p1 = v1.Position;
        glm::vec3 p2 = v2.Position;

        glm::vec2 uv0 = v0.UV;
        glm::vec2 uv1 = v1.UV;
        glm::vec2 uv2 = v2.UV;

        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;

        float u1 = uv1.x - uv0.x;
        float v1_ = uv1.y - uv0.y;
        float u2 = uv2.x - uv0.x;
        float v2_ = uv2.y - uv0.y;

        float det = (u1 * v2_ - u2 * v1_);
        float invDet = (det != 0.0f) ? 1.0f / det : 0.0f;

        glm::vec3 tangent = (edge1 * v2_ - edge2 * v1_) * invDet;
        glm::vec3 bitangent = (edge2 * u1 - edge1 * u2) * invDet;

        // Accumulate tangents and bitangents
        accumulatedTangents[i0] += tangent;
        accumulatedTangents[i1] += tangent;
        accumulatedTangents[i2] += tangent;

        accumulatedBitangents[i0] += bitangent;
        accumulatedBitangents[i1] += bitangent;
        accumulatedBitangents[i2] += bitangent;
    }

    // Normalize and orthogonalize
    for (size_t i = 0; i < vertices.size(); ++i) {
        glm::vec3 n = vertices[i].Normal;
        glm::vec3 t = accumulatedTangents[i];

        // Gram-Schmidt orthogonalization
        t = glm::normalize(t - glm::dot(n, t) * n);

        // Compute handedness via bitangent
        glm::vec3 b = glm::cross(n, t);
        glm::vec3 expectedB = accumulatedBitangents[i];

        // If the bitangent is pointing in the wrong direction, flip the tangent
        float handedness = (glm::dot(b, expectedB) < 0.0f) ? -1.0f : 1.0f;

        // Store the tangent and bitangent (bitangent only if you need it separately)
        vertices[i].Tangent = t * handedness;
        vertices[i].Bitangent = glm::cross(n, t) * handedness; // Optional, for use in shaders
    }
}

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

    // Create material buffer
    std::vector<RaytracingMaterial> rtMaterials;
    for (auto& material : Materials) {
        RaytracingMaterial mat = {};
        mat.AlbedoIndex = material.AlbedoView->GetDescriptor().Index;
        mat.NormalIndex = material.NormalView ? material.NormalView->GetDescriptor().Index : -1;
        mat.PBRIndex = material.PBRView ? material.PBRView->GetDescriptor().Index : -1;

        rtMaterials.push_back(mat);
    }

    MaterialBuffer = std::make_shared<Buffer>(sizeof(RaytracingMaterial) * rtMaterials.size(), sizeof(RaytracingMaterial), BufferType::Storage, "Material Buffer");
    MaterialBuffer->BuildSRV();

    Uploader::EnqueueBufferUpload(rtMaterials.data(), MaterialBuffer->GetSize(), MaterialBuffer);
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
    ComputeTangentSpace(vertices, indices);

    out.VertexCount = vertexCount;
    out.IndexCount = indexCount;

    /// @note(ame): create buffers
    out.VertexBuffer = std::make_shared<Buffer>(vertices.size() * sizeof(Vertex), sizeof(Vertex), BufferType::Storage, node->Name + " Vertex Buffer");
    out.VertexBuffer->BuildSRV();

    out.IndexBuffer = std::make_shared<Buffer>(indices.size() * sizeof(uint32_t), sizeof(uint32_t), BufferType::Storage, node->Name + " Index Buffer");
    out.IndexBuffer->BuildSRV();

    out.GeometryStructure = std::make_shared<BLAS>(out.VertexBuffer, out.IndexBuffer, out.VertexCount, out.IndexCount, node->Name + " BLAS");

    /// @note(ame): load and create textures
    cgltf_material *material = primitive->material;

    GLTFMaterial outMaterial = {};
    out.MaterialIndex = Materials.size();
    if (material) {
        if (material->pbr_metallic_roughness.base_color_texture.texture) {
            std::string path = Directory + '/' + std::string(material->pbr_metallic_roughness.base_color_texture.texture->image->uri);
    
            outMaterial.Albedo = TextureCache::Get(path);
            outMaterial.AlbedoView = std::make_shared<View>(outMaterial.Albedo, ViewType::ShaderResource, ViewDimension::Texture, TextureFormat::RGBA8_sRGB);
        } else {
            outMaterial.Albedo = RendererTools::Get("BlackTexture")->Texture;
            outMaterial.AlbedoView = RendererTools::Get("BlackTexture")->GetView(ViewType::ShaderResource);
        }

        if (material->normal_texture.texture) {
            std::string path = Directory + '/' + std::string(material->normal_texture.texture->image->uri);

            outMaterial.Normal = TextureCache::Get(path);
            outMaterial.NormalView = std::make_shared<View>(outMaterial.Normal, ViewType::ShaderResource, ViewDimension::Texture, TextureFormat::RGBA8);
        }

        if (material->pbr_metallic_roughness.metallic_roughness_texture.texture) {
            std::string path = Directory + '/' + std::string(material->pbr_metallic_roughness.metallic_roughness_texture.texture->image->uri);

            outMaterial.PBR = TextureCache::Get(path);
            outMaterial.PBRView = std::make_shared<View>(outMaterial.PBR, ViewType::ShaderResource, ViewDimension::Texture, TextureFormat::RGBA8);
        }

        outMaterial.AlphaTested = (material->alpha_mode != cgltf_alpha_mode_opaque);
    } else {
        outMaterial.Albedo = RendererTools::Get("BlackTexture")->Texture;
        outMaterial.AlbedoView = RendererTools::Get("BlackTexture")->GetView(ViewType::ShaderResource);
    }

    Materials.push_back(outMaterial);

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
