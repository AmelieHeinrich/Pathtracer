#include "gpu_internal.h"

#include <string.h>

// A hit group can carry up to three stages, so these two bounds are independent.
#define PT_MAX_RT_GROUPS 16
#define PT_MAX_RT_STAGES 32

static uint32_t align_up_u32(uint32_t value, uint32_t alignment)
{
    return alignment ? (value + alignment - 1) & ~(alignment - 1) : value;
}

// Packs the shader group handles into a shader binding table. The handle *size* and the
// per-entry *stride* are different numbers whenever handleSize != handleAlignment, and
// each region additionally starts on a shaderGroupBaseAlignment boundary.
static bool build_sbt(gpu_device_t *device, VkPipeline pipeline, uint32_t raygen_count,
                      uint32_t miss_count, uint32_t hit_count, gpu_sbt_t *out)
{
    const VkPhysicalDeviceRayTracingPipelinePropertiesKHR *props =
        &device->rt_pipeline_properties;

    const uint32_t handle_size = props->shaderGroupHandleSize;
    const uint32_t stride = align_up_u32(handle_size, props->shaderGroupHandleAlignment);
    const uint32_t base = props->shaderGroupBaseAlignment;

    const uint32_t raygen_size = align_up_u32(raygen_count * stride, base);
    const uint32_t miss_size = align_up_u32(miss_count * stride, base);
    const uint32_t hit_size = align_up_u32(hit_count * stride, base);

    const uint32_t group_count = raygen_count + miss_count + hit_count;
    uint8_t *handles = gpu_alloc(group_count * handle_size);
    const VkResult result = vkGetRayTracingShaderGroupHandlesKHR(
        device->device, pipeline, 0, group_count, group_count * handle_size, handles);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "gpu: vkGetRayTracingShaderGroupHandlesKHR failed: %s\n",
                vk_result_string(result));
        free(handles);
        return false;
    }

    out->buffer = gpu_buffer_create(device, raygen_size + miss_size + hit_size,
                                    VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                    GPU_MEMORY_UPLOAD);

    uint8_t *dst = out->buffer.mapped;
    const uint8_t *src = handles;
    memset(dst, 0, (size_t)out->buffer.size);

    uint32_t offset = 0;
    const uint32_t counts[3] = {raygen_count, miss_count, hit_count};
    const uint32_t sizes[3] = {raygen_size, miss_size, hit_size};
    VkStridedDeviceAddressRegionKHR *regions[3] = {&out->raygen, &out->miss, &out->hit};

    for (uint32_t region = 0; region < 3; ++region) {
        for (uint32_t i = 0; i < counts[region]; ++i) {
            memcpy(dst + offset + i * stride, src, handle_size);
            src += handle_size;
        }
        *regions[region] = (VkStridedDeviceAddressRegionKHR){
            .deviceAddress = out->buffer.address + offset,
            .stride = stride,
            .size = sizes[region],
        };
        offset += sizes[region];
    }

    // The raygen region is special: the spec requires size == stride, and the address must
    // sit on a shaderGroupBaseAlignment boundary. Both therefore take the base-aligned
    // region size, not the per-handle stride the miss and hit regions use.
    out->raygen.stride = out->raygen.size;
    out->callable = (VkStridedDeviceAddressRegionKHR){0};

    free(handles);
    return true;
}

bool gpu_rt_pipeline_create(gpu_device_t *device, gpu_uploader_t *uploader,
                            const gpu_rt_pipeline_desc_t *desc, gpu_rt_pipeline_t *out)
{
    (void)uploader;

    const uint32_t group_count = 1 + desc->miss_count + desc->hit_group_count;
    if (group_count > PT_MAX_RT_GROUPS) {
        fprintf(stderr, "gpu: %u shader groups exceeds PT_MAX_RT_GROUPS (%u)\n", group_count,
                PT_MAX_RT_GROUPS);
        return false;
    }
    // Worst case every hit group carries all three of its stages.
    if (1 + desc->miss_count + desc->hit_group_count * 3 > PT_MAX_RT_STAGES) {
        fprintf(stderr, "gpu: too many shader stages for PT_MAX_RT_STAGES (%u)\n",
                PT_MAX_RT_STAGES);
        return false;
    }

    // One module holds every entry point; the stages below select among them by name,
    // which only works because the shaders were compiled with -fvk-use-entrypoint-name.
    VkShaderModule module = gpu_shader_module_load(device, desc->spirv_path);
    if (!module) {
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[PT_MAX_RT_STAGES];
    VkRayTracingShaderGroupCreateInfoKHR groups[PT_MAX_RT_GROUPS];
    uint32_t stage_index = 0;
    uint32_t group_index = 0;

    // Group order must match the SBT region order: raygen, then miss, then hit.
    stages[stage_index] = (VkPipelineShaderStageCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        .module = module,
        .pName = desc->raygen_entry,
    };
    groups[group_index++] = (VkRayTracingShaderGroupCreateInfoKHR){
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
        .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
        .generalShader = stage_index,
        .closestHitShader = VK_SHADER_UNUSED_KHR,
        .anyHitShader = VK_SHADER_UNUSED_KHR,
        .intersectionShader = VK_SHADER_UNUSED_KHR,
    };
    ++stage_index;

    for (uint32_t i = 0; i < desc->miss_count; ++i) {
        stages[stage_index] = (VkPipelineShaderStageCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_MISS_BIT_KHR,
            .module = module,
            .pName = desc->miss_entries[i],
        };
        groups[group_index++] = (VkRayTracingShaderGroupCreateInfoKHR){
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            .generalShader = stage_index,
            .closestHitShader = VK_SHADER_UNUSED_KHR,
            .anyHitShader = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };
        ++stage_index;
    }

    // A hit group names up to three entry points; each becomes its own stage, so the group
    // records stage indices rather than a single one. Naming the same entry point from two
    // groups just creates two equivalent stages, which is exactly what the procedural
    // groups do with their shared closest hit shader.
    for (uint32_t i = 0; i < desc->hit_group_count; ++i) {
        const gpu_hit_group_t *hit = &desc->hit_groups[i];
        VkRayTracingShaderGroupCreateInfoKHR group = {
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            .type = hit->intersection ? VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
                                      : VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
            .generalShader = VK_SHADER_UNUSED_KHR,
            .closestHitShader = VK_SHADER_UNUSED_KHR,
            .anyHitShader = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };

        const struct {
            const char *entry;
            VkShaderStageFlagBits stage;
            uint32_t *slot;
        } members[] = {
            {hit->closest_hit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, &group.closestHitShader},
            {hit->any_hit, VK_SHADER_STAGE_ANY_HIT_BIT_KHR, &group.anyHitShader},
            {hit->intersection, VK_SHADER_STAGE_INTERSECTION_BIT_KHR, &group.intersectionShader},
        };
        for (uint32_t m = 0; m < PT_COUNT(members); ++m) {
            if (!members[m].entry) {
                continue;
            }
            stages[stage_index] = (VkPipelineShaderStageCreateInfo){
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = members[m].stage,
                .module = module,
                .pName = members[m].entry,
            };
            *members[m].slot = stage_index++;
        }

        groups[group_index++] = group;
    }

    gpu_rt_pipeline_t built = {0};

    const VkPushConstantRange push_range = {
        .stageFlags = GPU_RT_PUSH_CONSTANT_STAGES,
        .size = desc->push_constant_size,
    };
    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &desc->set_layout,
        .pushConstantRangeCount = desc->push_constant_size ? 1u : 0u,
        .pPushConstantRanges = desc->push_constant_size ? &push_range : NULL,
    };
    VK_CHECK(vkCreatePipelineLayout(device->device, &layout_info, NULL, &built.layout));

    const VkRayTracingPipelineCreateInfoKHR pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .stageCount = stage_index,
        .pStages = stages,
        .groupCount = group_index,
        .pGroups = groups,
        .maxPipelineRayRecursionDepth = desc->max_recursion,
        .layout = built.layout,
    };
    const VkResult result = vkCreateRayTracingPipelinesKHR(device->device, VK_NULL_HANDLE,
                                                           VK_NULL_HANDLE, 1, &pipeline_info,
                                                           NULL, &built.pipeline);
    vkDestroyShaderModule(device->device, module, NULL);

    if (result != VK_SUCCESS) {
        fprintf(stderr, "gpu: vkCreateRayTracingPipelinesKHR failed: %s\n",
                vk_result_string(result));
        vkDestroyPipelineLayout(device->device, built.layout, NULL);
        return false;
    }

    // One SBT record per *group*, not per stage.
    if (!build_sbt(device, built.pipeline, 1, desc->miss_count, desc->hit_group_count,
                   &built.sbt)) {
        vkDestroyPipeline(device->device, built.pipeline, NULL);
        vkDestroyPipelineLayout(device->device, built.layout, NULL);
        return false;
    }

    *out = built;
    return true;
}

void gpu_rt_pipeline_destroy(gpu_device_t *device, gpu_rt_pipeline_t *pipeline)
{
    gpu_buffer_destroy(device, &pipeline->sbt.buffer);
    if (pipeline->pipeline) {
        vkDestroyPipeline(device->device, pipeline->pipeline, NULL);
    }
    if (pipeline->layout) {
        vkDestroyPipelineLayout(device->device, pipeline->layout, NULL);
    }
    memset(pipeline, 0, sizeof(*pipeline));
}
