#pragma once

#include "scene.h" // for PT_MAX_NAME and pt_vertex_t; scene.h never includes this back

// The mesh library: every model tools/bake_assets.py has baked into assets/bin, loaded once at
// startup and held for the life of the process.
//
// There is no model parser here and there is deliberately not going to be one. A .ptm file is a
// header plus two blocks, the first of which is already an array of pt_vertex_t, so loading one
// is a validity check, two freads and two buffer uploads -- no format knowledge beyond what the
// header states, and no third party dependency.
//
// Every baked mesh is centred and uniformly scaled to fit [-1,1] in object space, which is the
// same convention the built-in shapes follow (see the comment at the top of scene.h). That is
// what lets a mesh be positioned, rotated and scaled by an ordinary instance transform, and what
// keeps the selection box and the gizmo correct for it without a single special case.
//
// The library is a module level singleton rather than something the scene owns, because
// pt_shape_from_name -- which scene_io.c calls while parsing, with no scene in hand -- has to be
// able to resolve a mesh's name. There is one scene in this application, so the distinction
// costs nothing.

#define PT_MAX_MESHES 32

typedef struct pt_mesh_t {
    char name[PT_MAX_NAME]; // the baked file's stem, which is also its shape name

    uint32_t vertex_count;
    uint32_t index_count;
    // Bounds of the baked (already normalised) vertices. Inside [-1,1], touching it on the
    // longest axis. A scene places a mesh on the ground with translation.y = -aabb_min[1] * scale.y.
    float aabb_min[3];
    float aabb_max[3];

    gpu_buffer_t vertices;
    gpu_buffer_t indices;
    gpu_accel_t blas;
} pt_mesh_t;

// Loads every .ptm in `dir`, in sorted order so a mesh's index is stable from run to run. A
// missing directory or an unreadable file warns and is skipped: no model should ever stop the
// renderer from starting. Calling this twice is a no-op.
void pt_mesh_library_init(gpu_device_t *device, gpu_uploader_t *uploader, const char *dir);
void pt_mesh_library_free(gpu_device_t *device);

uint32_t pt_mesh_count(void);
const pt_mesh_t *pt_mesh_get(uint32_t index);
