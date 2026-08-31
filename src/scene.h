#pragma once

#include "gpu.h"

#include <stdbool.h>

// The scene splits in two. The *authored* half -- entities, lights and the camera they were
// framed with -- is plain data that the UI edits and that src/scene_io.c round-trips through
// a text file. The *derived* half is the Vulkan objects built from it.
//
// The three bottom level structures are not derived: they are canonical unit shapes that
// never change, so they are built once and only the top level structure and the instance and
// light tables are rebuilt when the authored data moves.
//
// The analytic shapes are all defined as *canonical unit shapes in object space*, chosen so
// that each fits exactly inside [-1,-1,-1]..[1,1,1]:
//
//   sphere    |p| = 1
//   cylinder  x^2 + z^2 = 1, y in [-1,1], capped at y = +-1
//   cone      x^2 + z^2 = ((1-y)/2)^2, apex at y = +1, capped at y = -1
//
// So one BLAS over a single unit AABB backs all three, and an instance picks which shape it
// is purely through instanceShaderBindingTableRecordOffset. Position, orientation and size
// come from the instance transform, which is why the intersection shaders need no parameters.
//
// Baked models join the same scheme rather than sitting beside it. Every .ptm in assets/bin is
// centred and uniformly scaled into that same [-1,1] box by the baker, and registers itself as
// another *shape* at startup -- so `shape dragon` is written, parsed, listed and instanced by
// exactly the code that handles `shape cube`, and a mesh needs no new field on an entity, no
// second combo in the panel and no new key in the file format. See src/mesh.h.

#define PT_MAX_ENTITIES 256
#define PT_MAX_LIGHTS 32
#define PT_MAX_NAME 48

// ---------------------------------------------------------------------------
// authored data
// ---------------------------------------------------------------------------

// The analytic shapes, which are the only ones known at compile time. A shape *index* is
// wider than this enum: the baked meshes are appended after PT_SHAPE_BUILTIN_COUNT at startup,
// so the count to iterate to is pt_shape_count(), never PT_SHAPE_BUILTIN_COUNT.
typedef enum pt_shape_t {
    PT_SHAPE_PLANE = 0,
    PT_SHAPE_CUBE,
    PT_SHAPE_SPHERE,
    PT_SHAPE_CYLINDER,
    PT_SHAPE_CONE,
    PT_SHAPE_BUILTIN_COUNT,
} pt_shape_t;

// None of these has geometry in the acceleration structure, which is what lets the
// integrator sample them explicitly without ever double counting and without needing MIS:
// a BSDF ray simply cannot find one. The area light is drawn directly for the camera ray
// alone -- see the emission rule in shaders/pathtracer.slang -- which keeps that true while
// still letting you see the panel.
//
// A shape that glows is a different thing: an entity with a non-zero emission, found by BSDF
// rays only. Noisier, but it can be any shape.
typedef enum pt_light_type_t {
    PT_LIGHT_POINT = 0,
    PT_LIGHT_DIRECTIONAL,
    PT_LIGHT_SPOT,
    PT_LIGHT_AREA, // a one-sided rectangle emitting along `direction`
    PT_LIGHT_TYPE_COUNT,
} pt_light_type_t;

typedef struct pt_entity_t {
    char name[PT_MAX_NAME];
    pt_shape_t shape; // an index into the shape registry, so it may exceed the enum above

    float translation[3];
    float rotation[3]; // XYZ Euler, degrees
    float scale[3];

    float albedo[3];
    // Split the way a light's colour and intensity are, so the colour can be picked in [0,1]
    // while the brightness stays free to go as high as it likes.
    float emission[3];
    float emission_strength;
    float roughness; // 1 = fully diffuse, 0 = perfect mirror

    // 0 is a dielectric -- a coloured base under a white specular highlight. 1 is a conductor,
    // which has no diffuse lobe at all and tints its reflection with `albedo` instead.
    float metallic;
    // How much light passes through rather than reflecting. Dielectrics only; a conductor is
    // opaque by construction.
    float transmission;
    float ior; // refractive index at the sodium d-line, 587.6nm
    // Abbe number, the optical measure of how much the index varies across the visible band.
    // 0 disables dispersion for this material, which also lets the path keep all four of its
    // wavelengths instead of collapsing to one -- so it is meaningfully cheaper, not just
    // simpler. Real glasses run about 20 (dense flint, very dispersive) to 65 (crown).
    float abbe;
    // Beer-Lambert absorption, authored as the colour the material becomes at
    // `absorption_distance` of travel through it. White means no absorption at all.
    // fill_instance turns the pair into an extinction coefficient.
    float absorption[3];
    float absorption_distance;
} pt_entity_t;

typedef struct pt_light_t {
    char name[PT_MAX_NAME];
    pt_light_type_t type;

    float position[3];  // point and spot
    float direction[3]; // directional and spot; the direction the light points
    float color[3];
    float intensity;
    // Kelvin. 0 means "no temperature authored": the light keeps the colour it was given,
    // lit by the default illuminant. Non-zero multiplies a blackbody of that temperature in
    // on top, normalised so it changes the colour and never the brightness -- which is what
    // lets every scene written before this existed load unchanged.
    float temperature;

    // Past `range` the light contributes nothing, which bounds how many entities a shadow ray
    // has to be traced for. 0 means unbounded.
    float range;
    float cone_inner; // spot, degrees; full brightness inside this half angle
    float cone_outer; // spot, degrees; falls to zero at this half angle
    float size[2];    // area, the rectangle's half extents across and along
} pt_light_t;

// ---------------------------------------------------------------------------
// GPU mirrors
// ---------------------------------------------------------------------------

// Matches `struct Vertex` in shaders/pathtracer.slang under scalar layout. 24 bytes; the
// acceleration structure build reads only the leading position.
typedef struct pt_vertex_t {
    float position[3];
    float normal[3];
} pt_vertex_t;

// Matches `struct InstanceData` in shaders/pathtracer.slang under scalar layout. Indexed by
// InstanceIndex(), so it runs parallel to the TLAS instance array.
typedef struct pt_instance_data_t {
    VkDeviceAddress vertices; // 0 for the procedural shapes
    VkDeviceAddress indices;  // 0 for the procedural shapes
    float albedo[3];
    float emission[3];
    float roughness; // 1 = fully diffuse, 0 = perfect mirror
    float metallic;  // this is what the old explicit `pad` became
    float transmission;
    float ior;
    float abbe; // 0 disables dispersion
    // The authored colour and distance, folded into one: the fraction of light surviving a
    // *single unit* of travel. Beer-Lambert is then just this raised to the distance, so the
    // shader needs no logarithm -- and because it stays inside [0,1] it can be turned into a
    // spectrum by the same upsampling LUT as any other colour, which is what lets a tint
    // deepen with thickness the way a real one does.
    float attenuation[3];
} pt_instance_data_t;

// The device address members force 8 byte alignment, so this struct only ever grows in
// multiples of 8. Update this number deliberately, and move `struct InstanceData` in
// shaders/pathtracer.slang in the same commit -- a mismatch here corrupts every material
// silently rather than failing, which is exactly what this assert exists to prevent.
_Static_assert(sizeof(pt_instance_data_t) == 72,
               "pt_instance_data_t must match struct InstanceData in shaders/pathtracer.slang");

// Matches `struct Light` in shaders/pathtracer.slang under scalar layout. Cone angles arrive
// as cosines and the colour arrives premultiplied by intensity, so the shader does no trig
// and no extra multiply per sample. Every member here is 4 byte aligned, so unlike
// pt_instance_data_t -- whose device addresses push the struct alignment to 8 -- this needs
// no explicit padding to match.
typedef struct pt_light_data_t {
    uint32_t type;
    float position[3];
    float direction[3];
    float color[3]; // colour * intensity
    float range;    // 0 = unbounded
    float cos_inner;
    float cos_outer;
    float temperature; // Kelvin, or 0 for the default illuminant
    // Area lights: the rectangle's two half-edge vectors and its full area, derived here so
    // the shader never has to build a basis or take a cross product per sample.
    float edge_u[3];
    float edge_v[3];
    float area;
} pt_light_data_t;

// Every member is 4 byte aligned, so this one needs no explicit padding -- but that is a
// property worth failing the build over rather than rediscovering as a silent mismatch with
// struct Light in the shader.
_Static_assert(sizeof(pt_light_data_t) == 21 * sizeof(float),
               "pt_light_data_t must match struct Light in shaders/pathtracer.slang");

// Hit group indices, which are also the SBT record offsets carried by each instance. The
// order here is the order the hit groups are declared to the pipeline.
typedef enum pt_hit_group_t {
    PT_HIT_GROUP_MESH = 0,
    PT_HIT_GROUP_SPHERE,
    PT_HIT_GROUP_CYLINDER,
    PT_HIT_GROUP_CONE,
    PT_HIT_GROUP_COUNT,
} pt_hit_group_t;

typedef enum pt_blas_t {
    PT_BLAS_CUBE = 0,
    PT_BLAS_PLANE,
    PT_BLAS_UNIT_AABB, // shared by every procedural shape
    PT_BLAS_COUNT,
} pt_blas_t;

// ---------------------------------------------------------------------------
// scene
// ---------------------------------------------------------------------------

typedef struct pt_scene_t {
    // --- authored ---
    pt_entity_t entities[PT_MAX_ENTITIES];
    uint32_t entity_count;
    pt_light_t lights[PT_MAX_LIGHTS];
    uint32_t light_count;

    // The viewpoint the scene was framed from. The fly camera starts here and scene_io saves
    // wherever it ended up.
    float camera_position[3];
    float camera_yaw;   // degrees
    float camera_pitch; // degrees
    float camera_fov;   // vertical, degrees
    // Thin lens. 0 is a pinhole, which is what a scene file without these keys loads as, so
    // every scene authored before depth of field existed still renders exactly as it did.
    float camera_aperture;
    float camera_focus_distance;

    // Bumped by every edit. pt_scene_sync rebuilds when it moves and the renderer restarts
    // its accumulation on the same signal, so no edit path has to remember to do either.
    uint32_t revision;

    // --- static geometry, built once ---
    gpu_buffer_t mesh_vertices; // cube vertices, then plane vertices
    gpu_buffer_t mesh_indices;  // cube indices, then plane indices; each mesh-local
    gpu_buffer_t aabbs;         // one unit VkAabbPositionsKHR
    gpu_accel_t blas[PT_BLAS_COUNT];

    // --- derived, rebuilt by pt_scene_sync ---
    // Both tables are allocated at full capacity once and written through their persistent
    // mapping, so a sync never reallocates and never stages.
    gpu_buffer_t instance_data;
    gpu_buffer_t light_data;
    gpu_accel_t tlas;
    uint32_t synced_revision;
} pt_scene_t;

// Builds the static geometry and syncs whatever the authored arrays already hold. Fill those
// in (pt_scene_default, or scene_io) before calling.
void pt_scene_init(pt_scene_t *scene, gpu_device_t *device, gpu_uploader_t *uploader);
void pt_scene_free(pt_scene_t *scene, gpu_device_t *device);

// Rebuilds the top level structure and the instance and light tables from the authored data.
// A cheap no-op when the revision has not moved. Returns true if it rebuilt, which is the
// caller's cue to rewrite any descriptor pointing at the old TLAS.
bool pt_scene_sync(pt_scene_t *scene, gpu_device_t *device, gpu_uploader_t *uploader);

// Replaces the authored data with the built-in showcase. Used when no scene file loads.
void pt_scene_default(pt_scene_t *scene);

// Append and remove, all bumping the revision. The add functions return NULL when full.
pt_entity_t *pt_scene_add_entity(pt_scene_t *scene, pt_shape_t shape);
pt_light_t *pt_scene_add_light(pt_scene_t *scene, pt_light_type_t type);
void pt_scene_remove_entity(pt_scene_t *scene, uint32_t index);
void pt_scene_remove_light(pt_scene_t *scene, uint32_t index);

// Name <-> enum. One table each, shared by the UI and the file format so the two can never
// disagree about what a shape is called.
//
// The shape table is built at startup rather than being a constant, because the baked meshes
// extend it -- but only pt_scene_init writes it, so these stay ordinary lookups. Iterate to
// pt_shape_count(), which is the built-ins plus however many models were found.
uint32_t pt_shape_count(void);
const char *pt_shape_name(pt_shape_t shape);
bool pt_shape_from_name(const char *name, pt_shape_t *out);
const char *pt_light_type_name(pt_light_type_t type);
bool pt_light_type_from_name(const char *name, pt_light_type_t *out);
