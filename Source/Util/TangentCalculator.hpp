//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-08 21:17:06
//

#pragma once

#include <Oslo/Oslo.hpp>

#include <mikktspace/mikktspace.h>

struct Vertex
{
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 UV;
    glm::vec3 Tangent;
    glm::vec3 Bitangent;
};

struct MeshData
{
    std::vector<Vertex>* Vertices;
    std::vector<uint32_t>* Indices;
};

class TangentCalculator
{
public:
    TangentCalculator();

    void Calculate(MeshData* data);
private:
    SMikkTSpaceInterface iface{};
    SMikkTSpaceContext context{};

    static int get_vertex_index(const SMikkTSpaceContext *context, int iFace, int iVert);

    static int get_num_faces(const SMikkTSpaceContext *context);
    static int get_num_vertices_of_face(const SMikkTSpaceContext *context, int iFace);
    static void get_position(const SMikkTSpaceContext *context, float outpos[], int iFace, int iVert);
    static void get_normal(const SMikkTSpaceContext *context, float outnormal[], int iFace, int iVert);
    static void get_tex_coords(const SMikkTSpaceContext *context, float outuv[], int iFace, int iVert);
    static void set_tspace(const SMikkTSpaceContext * pContext, const float fvTangent[], const float fvBiTangent[], const float fMagS, const float fMagT, const tbool bIsOrientationPreserving, const int iFace, const int iVert);
};
