#include "scene.h"

#include "gpu_internal.h"
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

// One row per shape, so the name, the structure it instances and the hit group it selects can
// never disagree. The UI, the file format and pt_scene_sync all read it.
static const struct {
    const char *name;
    pt_blas_t blas;
    pt_hit_group_t hit_group;
} PT_SHAPES[PT_SHAPE_COUNT] = {
    [PT_SHAPE_PLANE] = {"plane", PT_BLAS_PLANE, PT_HIT_GROUP_MESH},
    [PT_SHAPE_CUBE] = {"cube", PT_BLAS_CUBE, PT_HIT_GROUP_MESH},
    [PT_SHAPE_SPHERE] = {"sphere", PT_BLAS_UNIT_AABB, PT_HIT_GROUP_SPHERE},
    [PT_SHAPE_CYLINDER] = {"cylinder", PT_BLAS_UNIT_AABB, PT_HIT_GROUP_CYLINDER},
    [PT_SHAPE_CONE] = {"cone", PT_BLAS_UNIT_AABB, PT_HIT_GROUP_CONE},
};

static const char *const PT_LIGHT_TYPE_NAMES[PT_LIGHT_TYPE_COUNT] = {
    [PT_LIGHT_POINT] = "point",
    [PT_LIGHT_DIRECTIONAL] = "directional",
    [PT_LIGHT_SPOT] = "spot",
    [PT_LIGHT_AREA] = "area",
};

const char *pt_shape_name(pt_shape_t shape)
{
    return shape < PT_SHAPE_COUNT ? PT_SHAPES[shape].name : "unknown";
}

bool pt_shape_from_name(const char *name, pt_shape_t *out)
{
    for (uint32_t i = 0; i < PT_SHAPE_COUNT; ++i) {
        if (strcmp(name, PT_SHAPES[i].name) == 0) {
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
    const pt_shape_t shape = entity->shape < PT_SHAPE_COUNT ? entity->shape : PT_SHAPE_CUBE;

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
        .instanceShaderBindingTableRecordOffset = (uint32_t)PT_SHAPES[shape].hit_group,
        .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
        .accelerationStructureReference = scene->blas[PT_SHAPES[shape].blas].address,
    };

    *data = (pt_instance_data_t){.roughness = entity->roughness};
    memcpy(data->albedo, entity->albedo, sizeof(data->albedo));
    // Premultiplied here, exactly as a light's colour and intensity are.
    for (uint32_t i = 0; i < 3; ++i) {
        data->emission[i] = entity->emission[i] * entity->emission_strength;
    }

    // Procedural shapes need no vertex data; their addresses stay null.
    if (PT_SHAPES[shape].blas == PT_BLAS_CUBE) {
        data->vertices = scene->mesh_vertices.address;
        data->indices = scene->mesh_indices.address;
    } else if (PT_SHAPES[shape].blas == PT_BLAS_PLANE) {
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

    for (uint32_t i = 0; i < 3; ++i) {
        data->color[i] = light->color[i] * light->intensity;
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
    memset(scene, 0, sizeof(*scene));
}
