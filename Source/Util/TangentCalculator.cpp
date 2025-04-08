//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-08 21:19:43
//

#include "TangentCalculator.hpp"

TangentCalculator::TangentCalculator()
{
    iface.m_getNumFaces = get_num_faces;
    iface.m_getNumVerticesOfFace = get_num_vertices_of_face;

    iface.m_getNormal = get_normal;
    iface.m_getPosition = get_position;
    iface.m_getTexCoord = get_tex_coords;
    iface.m_setTSpace = set_tspace;

    context.m_pInterface = &iface;
}

void TangentCalculator::Calculate(MeshData* data)
{
    context.m_pUserData = data;

    if (genTangSpaceDefault(&context) == false) {
        LOG_ERROR("Failed to generate tangents!");
    }
}

int TangentCalculator::get_num_faces(const SMikkTSpaceContext *context)
{
    MeshData *working_mesh = static_cast<MeshData*>(context->m_pUserData);
    return working_mesh->Indices->size();
}

int TangentCalculator::get_num_vertices_of_face(const SMikkTSpaceContext *context, const int iFace)
{
    return 3;
}

void TangentCalculator::get_position(const SMikkTSpaceContext *context, float *outpos, const int iFace, const int iVert)
{
    MeshData *working_mesh = static_cast<MeshData*>(context->m_pUserData);

    auto index = get_vertex_index(context, iFace, iVert);
    auto vertex = working_mesh->Vertices->at(index);

    outpos[0] = vertex.Position.x;
    outpos[1] = vertex.Position.y;
    outpos[2] = vertex.Position.z;
}

void TangentCalculator::get_normal(const SMikkTSpaceContext *context, float *outnormal, const int iFace, const int iVert)
{
    MeshData *working_mesh = static_cast<MeshData*>(context->m_pUserData);

    auto index = get_vertex_index(context, iFace, iVert);
    auto vertex = working_mesh->Vertices->at(index);

    outnormal[0] = vertex.Normal.x;
    outnormal[1] = vertex.Normal.y;
    outnormal[2] = vertex.Normal.z;
}

void TangentCalculator::get_tex_coords(const SMikkTSpaceContext *context, float *outuv, const int iFace, const int iVert)
{
    MeshData *working_mesh = static_cast<MeshData*>(context->m_pUserData);

    auto index = get_vertex_index(context, iFace, iVert);
    auto vertex = working_mesh->Vertices->at(index);

    outuv[0] = vertex.UV.x;
    outuv[1] = vertex.UV.y;
}

void TangentCalculator::set_tspace(const SMikkTSpaceContext * pContext, const float fvTangent[], const float fvBiTangent[], const float fMagS, const float fMagT, const tbool bIsOrientationPreserving, const int iFace, const int iVert)
{
    MeshData *working_mesh = static_cast<MeshData*>(pContext->m_pUserData);

    auto index = get_vertex_index(pContext, iFace, iVert);
    auto& vertex = working_mesh->Vertices->at(index);

    vertex.Tangent.x = fvTangent[0];
    vertex.Tangent.y = fvTangent[1];
    vertex.Tangent.z = fvTangent[2];
    vertex.Bitangent.x = fvBiTangent[0];
    vertex.Bitangent.y = fvBiTangent[1];
    vertex.Bitangent.z = fvBiTangent[2];
}

int TangentCalculator::get_vertex_index(const SMikkTSpaceContext *context, int iFace, int iVert) {
    MeshData *working_mesh = static_cast<MeshData*>(context->m_pUserData);

    auto face_size = get_num_vertices_of_face(context, iFace);

    auto indices_index = (iFace * face_size) + iVert;

    int index = working_mesh->Indices->at(indices_index);
    return index;
}
