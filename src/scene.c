#include "scene.h"

#include "spectral.h"

#include "gpu_internal.h"
#include "mesh.h"
#include "pt_math.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

// ---------------------------------------------------------------------------
// mesh data
// ---------------------------------------------------------------------------

#define PT_CUBE_VERTEX_COUNT 24
#define PT_CUBE_INDEX_COUNT 36
#define PT_PLANE_VERTEX_COUNT 4
#define PT_PLANE_INDEX_COUNT 6

// Both meshes share one vertex and one index buffer; each BLAS is built from an offset
// device address, so every mesh's indices stay 0-based within its own vertex range and
// PrimitiveIndex() * 3 indexes it directly in the closest hit shader.
//
// The cube is [-1,1]^3 with four vertices per face, so each face carries its own normal and
// stays flat. Faces run +X, -X, +Y, -Y, +Z, -Z.
static const pt_vertex_t PT_MESH_VERTICES[PT_CUBE_VERTEX_COUNT + PT_PLANE_VERTEX_COUNT] = {
    {{1.0f, -1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
    {{1.0f, -1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}},
    {{1.0f, 1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}},
    {{1.0f, 1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},

    {{-1.0f, -1.0f, -1.0f}, {-1.0f, 0.0f, 0.0f}},
    {{-1.0f, -1.0f, 1.0f}, {-1.0f, 0.0f, 0.0f}},
    {{-1.0f, 1.0f, 1.0f}, {-1.0f, 0.0f, 0.0f}},
    {{-1.0f, 1.0f, -1.0f}, {-1.0f, 0.0f, 0.0f}},

    {{-1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {{1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {{1.0f, 1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},
    {{-1.0f, 1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},

    {{-1.0f, -1.0f, -1.0f}, {0.0f, -1.0f, 0.0f}},
    {{1.0f, -1.0f, -1.0f}, {0.0f, -1.0f, 0.0f}},
    {{1.0f, -1.0f, 1.0f}, {0.0f, -1.0f, 0.0f}},
    {{-1.0f, -1.0f, 1.0f}, {0.0f, -1.0f, 0.0f}},

    {{-1.0f, -1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
    {{1.0f, -1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
    {{-1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},

    {{1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}},
    {{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}},
    {{-1.0f, 1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}},
    {{1.0f, 1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}},

    // Ground plane: a unit quad in XZ, scaled up by its instance transform.
    {{-1.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},
    {{-1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {{1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {{1.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},
};

static const uint32_t PT_MESH_INDICES[PT_CUBE_INDEX_COUNT + PT_PLANE_INDEX_COUNT] = {
    0,  1,  2,  0,  2,  3,  // +X
    4,  5,  6,  4,  6,  7,  // -X
    8,  9,  10, 8,  10, 11, // +Y
    12, 13, 14, 12, 14, 15, // -Y
    16, 17, 18, 16, 18, 19, // +Z
    20, 21, 22, 20, 22, 23, // -Z

    0, 1, 2, 0, 2, 3, // plane
};

#define PT_PLANE_VERTEX_OFFSET (PT_CUBE_VERTEX_COUNT * sizeof(pt_vertex_t))
#define PT_PLANE_INDEX_OFFSET (PT_CUBE_INDEX_COUNT * sizeof(uint32_t))

// Every procedural shape lives inside this box in object space.
static const VkAabbPositionsKHR PT_UNIT_AABB = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

// ---------------------------------------------------------------------------
// shape and light type tables
// ---------------------------------------------------------------------------

// One row per shape, so the name, the geometry it instances and the hit group it selects can
// never disagree. The UI, the file format and pt_scene_sync all read it.
//
// The analytic rows are statically initialised and the count starts at exactly them, so the
// table is complete and usable from program start -- pt_scene_default names its entities through
// pt_shape_name before pt_scene_init has run. register_mesh_shapes then appends one row per
// baked mesh, which is the whole of what makes a model a first class shape.
static struct {
    char name[PT_MAX_NAME];
    pt_blas_t blas;           // built-in rows only; a mesh carries its own structure
    pt_hit_group_t hit_group;
    int32_t mesh;             // -1 for a built-in, else an index into the mesh library
} g_shapes[PT_SHAPE_BUILTIN_COUNT + PT_MAX_MESHES] = {
    [PT_SHAPE_PLANE] = {"plane", PT_BLAS_PLANE, PT_HIT_GROUP_MESH, -1},
    [PT_SHAPE_CUBE] = {"cube", PT_BLAS_CUBE, PT_HIT_GROUP_MESH, -1},
    [PT_SHAPE_SPHERE] = {"sphere", PT_BLAS_UNIT_AABB, PT_HIT_GROUP_SPHERE, -1},
    [PT_SHAPE_CYLINDER] = {"cylinder", PT_BLAS_UNIT_AABB, PT_HIT_GROUP_CYLINDER, -1},
    [PT_SHAPE_CONE] = {"cone", PT_BLAS_UNIT_AABB, PT_HIT_GROUP_CONE, -1},
};
static uint32_t g_shape_count = PT_SHAPE_BUILTIN_COUNT;

// Appends the mesh library to the registry. Truncating back to the built-ins first makes this
// idempotent, and keeps every analytic shape's index exactly what its enumerator says -- which
// is what lets a scene file written before models existed still load unchanged.
static void register_mesh_shapes(void)
{
    g_shape_count = PT_SHAPE_BUILTIN_COUNT;

    for (uint32_t i = 0; i < pt_mesh_count(); ++i) {
        const pt_mesh_t *mesh = pt_mesh_get(i);

        // A model named after an existing shape would make pt_shape_from_name resolve to
        // whichever row it hit first, so the file quietly loses. Refuse it instead.
        pt_shape_t clash;
        if (pt_shape_from_name(mesh->name, &clash)) {
            fprintf(stderr, "scene: model '%s' collides with an existing shape, not registered\n",
                    mesh->name);
            continue;
        }

        snprintf(g_shapes[g_shape_count].name, sizeof(g_shapes[g_shape_count].name), "%s",
                 mesh->name);
        // A mesh is triangles, so it goes through the same hit group the cube and the plane do:
        // the shader reaches its vertices through the addresses fill_instance writes, and needs
        // to know nothing else about it.
        g_shapes[g_shape_count].hit_group = PT_HIT_GROUP_MESH;
        g_shapes[g_shape_count].mesh = (int32_t)i;
        ++g_shape_count;
    }
}

uint32_t pt_shape_count(void)
{
    return g_shape_count;
}

static const char *const PT_LIGHT_TYPE_NAMES[PT_LIGHT_TYPE_COUNT] = {
    [PT_LIGHT_POINT] = "point",
    [PT_LIGHT_DIRECTIONAL] = "directional",
    [PT_LIGHT_SPOT] = "spot",
    [PT_LIGHT_AREA] = "area",
};

const char *pt_shape_name(pt_shape_t shape)
{
    return shape < pt_shape_count() ? g_shapes[shape].name : "unknown";
}

bool pt_shape_from_name(const char *name, pt_shape_t *out)
{
    for (uint32_t i = 0; i < pt_shape_count(); ++i) {
        if (strcmp(name, g_shapes[i].name) == 0) {
            *out = (pt_shape_t)i;
            return true;
        }
    }
    return false;
}

const char *pt_light_type_name(pt_light_type_t type)
{
    return type < PT_LIGHT_TYPE_COUNT ? PT_LIGHT_TYPE_NAMES[type] : "unknown";
}

bool pt_light_type_from_name(const char *name, pt_light_type_t *out)
{
    for (uint32_t i = 0; i < PT_LIGHT_TYPE_COUNT; ++i) {
        if (strcmp(name, PT_LIGHT_TYPE_NAMES[i]) == 0) {
            *out = (pt_light_type_t)i;
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// authoring
// ---------------------------------------------------------------------------

void pt_scene_default(pt_scene_t *scene)
{
    scene->entity_count = 0;
    scene->light_count = 0;

    // The canonical shapes are centred on the origin and span [-1,1], so translating by
    // +scale.y in Y sets one down on the plane.
    struct {
        const char *name;
        pt_shape_t shape;
        float translation[3];
        float scale[3];
        float albedo[3];
        float roughness;
    } showcase[] = {
        {"ground", PT_SHAPE_PLANE, {0.0f, 0.0f, 0.0f}, {14.0f, 1.0f, 14.0f},
         {0.62f, 0.62f, 0.64f}, 1.0f},
        {"cube", PT_SHAPE_CUBE, {-3.3f, 0.8f, 0.0f}, {0.8f, 0.8f, 0.8f},
         {0.85f, 0.24f, 0.18f}, 1.0f},
        // Mirror, so the analytic sphere normals show up as a clean reflection.
        {"sphere", PT_SHAPE_SPHERE, {-1.1f, 0.85f, 0.0f}, {0.85f, 0.85f, 0.85f},
         {0.95f, 0.95f, 0.97f}, 0.0f},
        {"cylinder", PT_SHAPE_CYLINDER, {1.1f, 0.9f, 0.0f}, {0.55f, 0.9f, 0.55f},
         {0.24f, 0.68f, 0.34f}, 1.0f},
        {"cone", PT_SHAPE_CONE, {3.3f, 0.9f, 0.0f}, {0.75f, 0.9f, 0.75f},
         {0.26f, 0.42f, 0.9f}, 1.0f},
    };

    for (uint32_t i = 0; i < PT_COUNT(showcase); ++i) {
        pt_entity_t *entity = pt_scene_add_entity(scene, showcase[i].shape);
        snprintf(entity->name, sizeof(entity->name), "%s", showcase[i].name);
        memcpy(entity->translation, showcase[i].translation, sizeof(entity->translation));
        memcpy(entity->scale, showcase[i].scale, sizeof(entity->scale));
        memcpy(entity->albedo, showcase[i].albedo, sizeof(entity->albedo));
        entity->roughness = showcase[i].roughness;
    }

    // The overhead sphere that used to be an emissive instance is now an explicit light: the
    // integrator samples it directly, so it converges in a fraction of the frames.
    pt_light_t *key = pt_scene_add_light(scene, PT_LIGHT_POINT);
    snprintf(key->name, sizeof(key->name), "key");
    key->position[0] = 0.5f;
    key->position[1] = 4.4f;
    key->position[2] = 2.6f;
    key->color[0] = 1.0f;
    key->color[1] = 0.92f;
    key->color[2] = 0.78f;
    key->intensity = 60.0f;

    const float position[3] = {4.6f, 3.1f, 8.6f};
    memcpy(scene->camera_position, position, sizeof(scene->camera_position));
    // Framed on the row of primitives, matching the fixed viewpoint this scene used to have.
    scene->camera_yaw = -152.0f;
    scene->camera_pitch = -13.0f;
    scene->camera_fov = 40.0f;
    // A pinhole by default: depth of field is something to reach for, not something to
    // discover having been applied.
    scene->camera_aperture = 0.0f;
    scene->camera_focus_distance = 10.0f;

    ++scene->revision;
}

pt_entity_t *pt_scene_add_entity(pt_scene_t *scene, pt_shape_t shape)
{
    if (scene->entity_count >= PT_MAX_ENTITIES) {
        fprintf(stderr, "scene: entity limit (%u) reached\n", PT_MAX_ENTITIES);
        return NULL;
    }

    pt_entity_t *entity = &scene->entities[scene->entity_count++];
    memset(entity, 0, sizeof(*entity));
    entity->shape = shape;
    snprintf(entity->name, sizeof(entity->name), "%s", pt_shape_name(shape));
    entity->scale[0] = 1.0f;
    entity->scale[1] = 1.0f;
    entity->scale[2] = 1.0f;
    entity->albedo[0] = 0.7f;
    entity->albedo[1] = 0.7f;
    entity->albedo[2] = 0.7f;
    // White at unit strength, so an entity only glows once someone gives it a colour. Also
    // what makes a scene file written before emission was split load unchanged: back then the
    // colour *was* the full value, and multiplying it by one reproduces exactly that.
    entity->emission[0] = 1.0f;
    entity->emission[1] = 1.0f;
    entity->emission[2] = 1.0f;
    entity->emission_strength = 0.0f;
    entity->roughness = 1.0f;
    // Only the two that must not be zero. A zeroed absorption colour would mean *infinite*
    // extinction -- instantly black glass -- and a zeroed index would refract nonsensically,
    // so both are set here for the same reason emission defaults to white above.
    entity->ior = 1.5f;
    entity->absorption[0] = 1.0f;
    entity->absorption[1] = 1.0f;
    entity->absorption[2] = 1.0f;
    entity->absorption_distance = 1.0f;

    ++scene->revision;
    return entity;
}

pt_light_t *pt_scene_add_light(pt_scene_t *scene, pt_light_type_t type)
{
    if (scene->light_count >= PT_MAX_LIGHTS) {
        fprintf(stderr, "scene: light limit (%u) reached\n", PT_MAX_LIGHTS);
        return NULL;
    }

    pt_light_t *light = &scene->lights[scene->light_count++];
    memset(light, 0, sizeof(*light));
    light->type = type;
    snprintf(light->name, sizeof(light->name), "%s", pt_light_type_name(type));
    light->position[1] = 4.0f;
    light->direction[1] = -1.0f; // pointing down
    light->color[0] = 1.0f;
    light->color[1] = 1.0f;
    light->color[2] = 1.0f;
    light->intensity = type == PT_LIGHT_DIRECTIONAL ? 3.0f : 60.0f;
    light->cone_inner = 20.0f;
    light->cone_outer = 30.0f;
    light->size[0] = 0.5f;
    light->size[1] = 0.5f;
    if (type == PT_LIGHT_AREA) {
        // An area light's brightness is radiance over its whole face, so the numbers that
        // suit a point light's inverse-square falloff are wildly too large here.
        light->intensity = 8.0f;
    }

    ++scene->revision;
    return light;
}

void pt_scene_remove_entity(pt_scene_t *scene, uint32_t index)
{
    if (index >= scene->entity_count) {
        return;
    }
    // Order is what the UI list shows, so close the gap rather than swapping with the last.
    memmove(&scene->entities[index], &scene->entities[index + 1],
            (scene->entity_count - index - 1) * sizeof(scene->entities[0]));
    --scene->entity_count;
    ++scene->revision;
}

void pt_scene_remove_light(pt_scene_t *scene, uint32_t index)
{
    if (index >= scene->light_count) {
        return;
    }
    memmove(&scene->lights[index], &scene->lights[index + 1],
            (scene->light_count - index - 1) * sizeof(scene->lights[0]));
    --scene->light_count;
    ++scene->revision;
}

// ---------------------------------------------------------------------------
// build and sync
// ---------------------------------------------------------------------------

void pt_scene_init(pt_scene_t *scene, gpu_device_t *device, gpu_uploader_t *uploader)
{
    // Before any geometry of our own: the baked models, so that they are registered as shapes
    // by the time anything asks to resolve a shape name. renderer_init runs this well before
    // main.c loads a scene file, which is what lets that file say `shape dragon`.
    pt_mesh_library_init(device, uploader, PT_ASSET_DIR "/bin");
    register_mesh_shapes();

    // SHADER_DEVICE_ADDRESS so the shader can reach these as pointers; the AS build input
    // bit so they can also feed the geometry build directly.
    const VkBufferUsageFlags geometry_usage =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    scene->mesh_vertices = gpu_buffer_create(device, sizeof(PT_MESH_VERTICES), geometry_usage,
                                             GPU_MEMORY_DEVICE);
    scene->mesh_indices = gpu_buffer_create(device, sizeof(PT_MESH_INDICES), geometry_usage,
                                            GPU_MEMORY_DEVICE);
    scene->aabbs = gpu_buffer_create(device, sizeof(PT_UNIT_AABB), geometry_usage,
                                     GPU_MEMORY_DEVICE);

    gpu_buffer_upload(uploader, &scene->mesh_vertices, PT_MESH_VERTICES,
                      sizeof(PT_MESH_VERTICES));
    gpu_buffer_upload(uploader, &scene->mesh_indices, PT_MESH_INDICES, sizeof(PT_MESH_INDICES));
    gpu_buffer_upload(uploader, &scene->aabbs, &PT_UNIT_AABB, sizeof(PT_UNIT_AABB));

    scene->blas[PT_BLAS_CUBE] =
        gpu_blas_build_triangles(device, uploader, scene->mesh_vertices.address,
                                 PT_CUBE_VERTEX_COUNT, sizeof(pt_vertex_t),
                                 scene->mesh_indices.address, PT_CUBE_INDEX_COUNT);
    scene->blas[PT_BLAS_PLANE] = gpu_blas_build_triangles(
        device, uploader, scene->mesh_vertices.address + PT_PLANE_VERTEX_OFFSET,
        PT_PLANE_VERTEX_COUNT, sizeof(pt_vertex_t),
        scene->mesh_indices.address + PT_PLANE_INDEX_OFFSET, PT_PLANE_INDEX_COUNT);
    scene->blas[PT_BLAS_UNIT_AABB] =
        gpu_blas_build_aabbs(device, uploader, scene->aabbs.address, 1);

    // Allocated once at full capacity and host visible, so editing the scene is a memcpy
    // through the mapping with no reallocation and no staging.
    scene->instance_data =
        gpu_buffer_create(device, PT_MAX_ENTITIES * sizeof(pt_instance_data_t),
                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          GPU_MEMORY_UPLOAD);
    scene->light_data = gpu_buffer_create(device, PT_MAX_LIGHTS * sizeof(pt_light_data_t),
                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          GPU_MEMORY_UPLOAD);

    // Forces the first sync regardless of what the authored data's revision happens to be.
    scene->synced_revision = scene->revision - 1;
    pt_scene_sync(scene, device, uploader);
}

// A non-finite instance transform faults the GPU outright rather than merely rendering
// wrongly, so nothing is trusted to have produced a sane one. The gizmo already refuses to
// generate garbage, but a hand-edited scene file is not under our control at all.
static bool transform_is_finite(const VkTransformMatrixKHR *transform)
{
    for (uint32_t row = 0; row < 3; ++row) {
        for (uint32_t col = 0; col < 4; ++col) {
            if (!isfinite(transform->matrix[row][col])) {
                return false;
            }
        }
    }
    return true;
}

// Turns one authored entity into the TLAS instance and the material record that must sit at
// the same index in both tables.
static void fill_instance(const pt_scene_t *scene, const pt_entity_t *entity,
                          VkAccelerationStructureInstanceKHR *instance,
                          pt_instance_data_t *data)
{
    // A scene file naming a model that has not been baked leaves an out of range index here,
    // so this clamp is what stops it from indexing the registry off its end.
    const pt_shape_t shape = entity->shape < pt_shape_count() ? entity->shape : PT_SHAPE_CUBE;
    const pt_mesh_t *mesh = g_shapes[shape].mesh >= 0
                                ? pt_mesh_get((uint32_t)g_shapes[shape].mesh)
                                : NULL;

    VkTransformMatrixKHR transform =
        pt_transform_compose(entity->translation, entity->rotation, entity->scale);
    if (!transform_is_finite(&transform)) {
        fprintf(stderr, "scene: entity '%s' has a non-finite transform, using identity\n",
                entity->name);
        transform = (VkTransformMatrixKHR){.matrix = {{1.0f, 0.0f, 0.0f, 0.0f},
                                                      {0.0f, 1.0f, 0.0f, 0.0f},
                                                      {0.0f, 0.0f, 1.0f, 0.0f}}};
    }

    *instance = (VkAccelerationStructureInstanceKHR){
        .transform = transform,
        .mask = 0xFF,
        .instanceShaderBindingTableRecordOffset = (uint32_t)g_shapes[shape].hit_group,
        .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
        // A mesh brings its own structure; the analytic shapes share the scene's three.
        .accelerationStructureReference = mesh ? mesh->blas.address
                                              : scene->blas[g_shapes[shape].blas].address,
    };

    *data = (pt_instance_data_t){
        .roughness = entity->roughness,
        .metallic = entity->metallic,
        .transmission = entity->transmission,
        .ior = entity->ior,
        .abbe = entity->abbe,
    };
    memcpy(data->albedo, entity->albedo, sizeof(data->albedo));

    // Beer-Lambert, authored as "this colour after travelling this far" and re-expressed here
    // as "this colour after travelling one unit" -- the same place a light's intensity and
    // blackbody scale are folded in, and for the same reason.
    //
    // Transmittance is multiplicative in distance, so colour = unit^distance and the unit
    // value is the distance'th root. That leaves the shader a single pow, and leaves the
    // value inside [0,1] where the spectral upsampling LUT is defined. A white colour gives
    // exactly one, which is what keeps absorption inert until someone asks for it.
    const float distance = entity->absorption_distance > 1e-4f ? entity->absorption_distance
                                                               : 1e-4f;
    for (uint32_t i = 0; i < 3; ++i) {
        // Floored above zero: a pure black channel absorbs infinitely, and its root would
        // reach the shader as a zero that no amount of thickness could ever recover from.
        const float channel = fminf(fmaxf(entity->absorption[i], 1e-4f), 1.0f);
        data->attenuation[i] = powf(channel, 1.0f / distance);
    }
    // Premultiplied here, exactly as a light's colour and intensity are.
    for (uint32_t i = 0; i < 3; ++i) {
        data->emission[i] = entity->emission[i] * entity->emission_strength;
    }

    // Procedural shapes need no vertex data; their addresses stay null. Everything that does
    // reach closesthit_mesh gets its vertices and indices as raw device addresses, which is why
    // a baked model needs nothing the cube did not already need.
    if (mesh) {
        data->vertices = mesh->vertices.address;
        data->indices = mesh->indices.address;
    } else if (g_shapes[shape].blas == PT_BLAS_CUBE) {
        data->vertices = scene->mesh_vertices.address;
        data->indices = scene->mesh_indices.address;
    } else if (g_shapes[shape].blas == PT_BLAS_PLANE) {
        data->vertices = scene->mesh_vertices.address + PT_PLANE_VERTEX_OFFSET;
        data->indices = scene->mesh_indices.address + PT_PLANE_INDEX_OFFSET;
    }
}

static void fill_light(const pt_light_t *light, pt_light_data_t *data)
{
    *data = (pt_light_data_t){
        .type = (uint32_t)light->type,
        .range = light->range,
        // Cosines, so the shader compares against dot products directly. Note the ordering:
        // a *larger* cosine is a *smaller* angle, so cos_inner >= cos_outer.
        .cos_inner = cosf(light->cone_inner * 3.14159265358979323846f / 180.0f),
        .cos_outer = cosf(light->cone_outer * 3.14159265358979323846f / 180.0f),
    };
    memcpy(data->position, light->position, sizeof(data->position));

    // Normalised here so the shader never has to, and defaulted to straight down rather than
    // left as a zero vector a directional light could not be sampled from.
    memcpy(data->direction, light->direction, sizeof(data->direction));
    if (pt_vec3_length(data->direction) < 1e-6f) {
        data->direction[0] = 0.0f;
        data->direction[1] = -1.0f;
        data->direction[2] = 0.0f;
    }
    pt_vec3_normalize(data->direction);

    // The blackbody normalisation is folded in here, exactly as the intensity is: the shader
    // then needs no extra multiply, and a temperature changes only the light's colour because
    // the scale it comes with holds its luminance at 1.
    data->temperature = light->temperature;
    const float norm = pt_spectral_blackbody_norm(light->temperature);
    for (uint32_t i = 0; i < 3; ++i) {
        data->color[i] = light->color[i] * light->intensity * norm;
    }

    if (light->type != PT_LIGHT_AREA) {
        return;
    }

    // The rectangle's own orientation about its normal is not authored, so it is derived --
    // any consistent choice will do, since a rectangle rotated in its own plane emits
    // identically. The axis `direction` is least aligned with keeps the cross well
    // conditioned, the same trick the integrator uses to build a tangent.
    const float helper[3] = {fabsf(data->direction[0]) > 0.9f ? 0.0f : 1.0f,
                             fabsf(data->direction[0]) > 0.9f ? 1.0f : 0.0f, 0.0f};
    float u[3];
    float v[3];
    pt_vec3_cross(u, data->direction, helper);
    pt_vec3_normalize(u);
    pt_vec3_cross(v, data->direction, u);

    pt_vec3_scale(data->edge_u, u, light->size[0]);
    pt_vec3_scale(data->edge_v, v, light->size[1]);
    data->area = 4.0f * light->size[0] * light->size[1];
}

bool pt_scene_sync(pt_scene_t *scene, gpu_device_t *device, gpu_uploader_t *uploader)
{
    if (scene->revision == scene->synced_revision) {
        return false;
    }

    // Before anything else: the instance and light tables are host visible and are read
    // directly by shaders through device addresses, so writing them while a frame is still
    // in flight is a data race on memory the GPU is dereferencing -- and the structure freed
    // below could still be being traced against.
    VK_CHECK(vkDeviceWaitIdle(device->device));

    VkAccelerationStructureInstanceKHR instances[PT_MAX_ENTITIES];
    pt_instance_data_t *data = scene->instance_data.mapped;

    // The two arrays run index for index, which is what lets the shader look a material up
    // with InstanceIndex() and leaves instanceCustomIndex free.
    for (uint32_t i = 0; i < scene->entity_count; ++i) {
        fill_instance(scene, &scene->entities[i], &instances[i], &data[i]);
    }

    pt_light_data_t *lights = scene->light_data.mapped;
    for (uint32_t i = 0; i < scene->light_count; ++i) {
        fill_light(&scene->lights[i], &lights[i]);
    }

    gpu_accel_destroy(device, &scene->tlas);
    scene->tlas = gpu_tlas_build(device, uploader, instances, scene->entity_count);

    scene->synced_revision = scene->revision;
    return true;
}

void pt_scene_free(pt_scene_t *scene, gpu_device_t *device)
{
    gpu_accel_destroy(device, &scene->tlas);
    for (uint32_t i = 0; i < PT_BLAS_COUNT; ++i) {
        gpu_accel_destroy(device, &scene->blas[i]);
    }
    gpu_buffer_destroy(device, &scene->light_data);
    gpu_buffer_destroy(device, &scene->instance_data);
    gpu_buffer_destroy(device, &scene->aabbs);
    gpu_buffer_destroy(device, &scene->mesh_indices);
    gpu_buffer_destroy(device, &scene->mesh_vertices);
    // Symmetric with pt_scene_init, which loaded it. The library outlives any one scene in
    // principle, but this application has exactly one, so it is torn down with it.
    pt_mesh_library_free(device);
    register_mesh_shapes(); // drops the mesh rows, leaving the built-ins intact
    memset(scene, 0, sizeof(*scene));
}
