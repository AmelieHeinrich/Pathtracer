#include "gpu_internal.h"

// Reads a whole file into a gpu_alloc'd buffer. Returns NULL (and logs) on failure rather
// than aborting, so a missing or broken .spv cannot take the process down during reload.
static uint32_t *read_spirv(const char *path, size_t *out_size)
{
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "gpu: cannot open shader '%s'\n", path);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    const long size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (size <= 0 || (size % 4) != 0) {
        fprintf(stderr, "gpu: '%s' is not a valid SPIR-V module (%ld bytes)\n", path, size);
        fclose(file);
        return NULL;
    }

    uint32_t *code = gpu_alloc((size_t)size);
    if (fread(code, 1, (size_t)size, file) != (size_t)size) {
        fprintf(stderr, "gpu: short read on '%s'\n", path);
        free(code);
        fclose(file);
        return NULL;
    }
    fclose(file);

    *out_size = (size_t)size;
    return code;
}

VkShaderModule gpu_shader_module_load(gpu_device_t *device, const char *path)
{
    size_t code_size = 0;
    uint32_t *code = read_spirv(path, &code_size);
    if (!code) {
        return VK_NULL_HANDLE;
    }

    const VkShaderModuleCreateInfo module_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code_size,
        .pCode = code,
    };
    VkShaderModule module = VK_NULL_HANDLE;
    const VkResult result = vkCreateShaderModule(device->device, &module_info, NULL, &module);
    free(code);

    if (result != VK_SUCCESS) {
        fprintf(stderr, "gpu: vkCreateShaderModule failed for '%s': %s\n", path,
                vk_result_string(result));
        return VK_NULL_HANDLE;
    }
    return module;
}
