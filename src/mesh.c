#include "mesh.h"

#include "gpu_internal.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The on-disk header, written by tools/bake_assets.py. Every member is 4 byte aligned and the
// fields happen to sum to exactly 128, so this needs no packing attribute and no padding member
// -- the static assert below is what keeps that true if anyone ever adds a field.
typedef struct ptm_header_t {
    char magic[8]; // "PTMESH01", deliberately not NUL terminated
    uint32_t version;
    uint32_t vertex_count;
    uint32_t index_count;
    float aabb_min[3];
    float aabb_max[3];
    float source_center[3]; // what the baker subtracted before scaling
    float source_scale;     // the uniform scale it applied
    uint32_t flags;
    char name[64];
} ptm_header_t;

_Static_assert(sizeof(ptm_header_t) == 128, "the .ptm header must match the baker byte for byte");
_Static_assert(sizeof(pt_vertex_t) == 24, "the .ptm vertex block is an array of pt_vertex_t");

#define PTM_MAGIC "PTMESH01"
#define PTM_VERSION 1u

static pt_mesh_t g_meshes[PT_MAX_MESHES];
static uint32_t g_mesh_count;
static bool g_loaded;

uint32_t pt_mesh_count(void)
{
    return g_mesh_count;
}

const pt_mesh_t *pt_mesh_get(uint32_t index)
{
    return index < g_mesh_count ? &g_meshes[index] : NULL;
}

// Reads one baked mesh and turns it into two buffers and a bottom level structure. Everything
// that can be wrong with the file is a warning and a `false`, never an abort: a corrupt model
// must not be able to stop the renderer from starting.
static bool load_mesh(pt_mesh_t *mesh, gpu_device_t *device, gpu_uploader_t *uploader,
                      const char *path, const char *name)
{
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "mesh: cannot open '%s'\n", path);
        return false;
    }

    ptm_header_t header;
    if (fread(&header, sizeof(header), 1, file) != 1) {
        fprintf(stderr, "mesh: '%s' is too short to hold a header\n", path);
        fclose(file);
        return false;
    }

    if (memcmp(header.magic, PTM_MAGIC, sizeof(header.magic)) != 0) {
        fprintf(stderr, "mesh: '%s' is not a .ptm file\n", path);
        fclose(file);
        return false;
    }
    if (header.version != PTM_VERSION) {
        fprintf(stderr, "mesh: '%s' is version %u, this build reads %u -- re-run "
                        "tools/bake_assets.py --force\n",
                path, header.version, PTM_VERSION);
        fclose(file);
        return false;
    }
    if (header.vertex_count == 0 || header.index_count == 0 || header.index_count % 3 != 0) {
        fprintf(stderr, "mesh: '%s' has a nonsensical vertex/index count (%u/%u)\n", path,
                header.vertex_count, header.index_count);
        fclose(file);
        return false;
    }

    const size_t vertex_bytes = (size_t)header.vertex_count * sizeof(pt_vertex_t);
    const size_t index_bytes = (size_t)header.index_count * sizeof(uint32_t);

    // The file must be exactly the size its own header claims. Checked before either read, so a
    // truncated file is rejected rather than leaving half a mesh in a buffer -- and so a header
    // claiming a wild count cannot ask for a wild allocation either.
    if (fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "mesh: cannot seek '%s'\n", path);
        fclose(file);
        return false;
    }
    const long size = ftell(file);
    if (size < 0 || (size_t)size != sizeof(header) + vertex_bytes + index_bytes) {
        fprintf(stderr, "mesh: '%s' is %ld bytes, its header describes %zu\n", path, size,
                sizeof(header) + vertex_bytes + index_bytes);
        fclose(file);
        return false;
    }
    if (fseek(file, (long)sizeof(header), SEEK_SET) != 0) {
        fprintf(stderr, "mesh: cannot seek '%s'\n", path);
        fclose(file);
        return false;
    }

    void *vertices = gpu_alloc(vertex_bytes);
    void *indices = gpu_alloc(index_bytes);
    const bool read_ok = fread(vertices, vertex_bytes, 1, file) == 1 &&
                         fread(indices, index_bytes, 1, file) == 1;
    fclose(file);

    if (!read_ok) {
        fprintf(stderr, "mesh: failed to read the geometry out of '%s'\n", path);
        free(vertices);
        free(indices);
        return false;
    }

    // The same usage the built-in geometry uses in pt_scene_init: a device address so the
    // closest hit shader can walk it as a pointer, and the build input bit so the same memory
    // can feed the structure build directly.
    const VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    memset(mesh, 0, sizeof(*mesh));
    snprintf(mesh->name, sizeof(mesh->name), "%s", name);
    mesh->vertex_count = header.vertex_count;
    mesh->index_count = header.index_count;
    memcpy(mesh->aabb_min, header.aabb_min, sizeof(mesh->aabb_min));
    memcpy(mesh->aabb_max, header.aabb_max, sizeof(mesh->aabb_max));

    mesh->vertices = gpu_buffer_create(device, vertex_bytes, usage, GPU_MEMORY_DEVICE);
    mesh->indices = gpu_buffer_create(device, index_bytes, usage, GPU_MEMORY_DEVICE);
    gpu_buffer_upload(uploader, &mesh->vertices, vertices, vertex_bytes);
    gpu_buffer_upload(uploader, &mesh->indices, indices, index_bytes);

    free(vertices);
    free(indices);

    mesh->blas = gpu_blas_build_triangles(device, uploader, mesh->vertices.address,
                                          mesh->vertex_count, sizeof(pt_vertex_t),
                                          mesh->indices.address, mesh->index_count);

    printf("mesh: %s (%u vertices, %u triangles)\n", mesh->name, mesh->vertex_count,
           mesh->index_count / 3);
    fflush(stdout);
    return true;
}

// Sorted, so a mesh's index -- and therefore the shape index the scene registry hands it -- is
// the same on every run. Scene files reference a mesh by name, so this is not load bearing, but
// an index that silently reshuffles when a file is added is a debugging trap.
static int compare_names(const void *a, const void *b)
{
    return strcmp((const char *)a, (const char *)b);
}

void pt_mesh_library_init(gpu_device_t *device, gpu_uploader_t *uploader, const char *dir)
{
    if (g_loaded) {
        return;
    }
    g_loaded = true;

    DIR *directory = opendir(dir);
    if (!directory) {
        fprintf(stderr, "mesh: no asset directory '%s' -- run tools/bake_assets.py\n", dir);
        return;
    }

    char names[PT_MAX_MESHES][PT_MAX_NAME];
    uint32_t name_count = 0;

    const struct dirent *entry;
    while ((entry = readdir(directory)) != NULL) {
        const size_t length = strlen(entry->d_name);
        if (length < 5 || strcmp(entry->d_name + length - 4, ".ptm") != 0) {
            continue;
        }
        if (name_count >= PT_MAX_MESHES) {
            fprintf(stderr, "mesh: more than %u models in '%s', ignoring the rest\n",
                    PT_MAX_MESHES, dir);
            break;
        }
        // The stem is the shape name, so it has to fit in a name buffer with room for its
        // terminator; anything longer would be truncated into a collision with its neighbours.
        if (length - 4 >= PT_MAX_NAME) {
            fprintf(stderr, "mesh: '%s' has too long a name, skipped\n", entry->d_name);
            continue;
        }
        memcpy(names[name_count], entry->d_name, length - 4);
        names[name_count][length - 4] = '\0';
        ++name_count;
    }
    closedir(directory);

    qsort(names, name_count, sizeof(names[0]), compare_names);

    for (uint32_t i = 0; i < name_count; ++i) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s.ptm", dir, names[i]);
        if (load_mesh(&g_meshes[g_mesh_count], device, uploader, path, names[i])) {
            ++g_mesh_count;
        }
    }
}

void pt_mesh_library_free(gpu_device_t *device)
{
    for (uint32_t i = 0; i < g_mesh_count; ++i) {
        gpu_accel_destroy(device, &g_meshes[i].blas);
        gpu_buffer_destroy(device, &g_meshes[i].indices);
        gpu_buffer_destroy(device, &g_meshes[i].vertices);
    }
    memset(g_meshes, 0, sizeof(g_meshes));
    g_mesh_count = 0;
    g_loaded = false;
}
