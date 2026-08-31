#include "gpu_internal.h"

#include <string.h>

static VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize alignment)
{
    return alignment ? (value + alignment - 1) & ~(alignment - 1) : value;
}

gpu_accel_t gpu_accel_build(gpu_device_t *device, gpu_uploader_t *uploader,
                            VkAccelerationStructureTypeKHR type,
                            const VkAccelerationStructureGeometryKHR *geometries,
                            const uint32_t *primitive_counts, uint32_t geometry_count)
{
    gpu_accel_t accel = {0};

    VkAccelerationStructureBuildGeometryInfoKHR build = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = type,
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .geometryCount = geometry_count,
        .pGeometries = geometries,
    };

    VkAccelerationStructureBuildSizesInfoKHR sizes = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    };
    vkGetAccelerationStructureBuildSizesKHR(device->device,
                                            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &build, primitive_counts, &sizes);

    accel.buffer = gpu_buffer_create(device, sizes.accelerationStructureSize,
                                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                     GPU_MEMORY_DEVICE);

    const VkAccelerationStructureCreateInfoKHR create_info = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .buffer = accel.buffer.handle,
        .size = sizes.accelerationStructureSize,
        .type = type,
    };
    VK_CHECK(vkCreateAccelerationStructureKHR(device->device, &create_info, NULL,
                                              &accel.handle));

    // Scratch has its own alignment requirement, distinct from the buffer's own.
    const VkDeviceSize scratch_alignment =
        device->accel_properties.minAccelerationStructureScratchOffsetAlignment;
    gpu_buffer_t scratch = gpu_buffer_create(
        device, align_up(sizes.buildScratchSize, scratch_alignment) + scratch_alignment,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        GPU_MEMORY_DEVICE);

    build.dstAccelerationStructure = accel.handle;
    build.scratchData.deviceAddress = align_up(scratch.address, scratch_alignment);

    VkAccelerationStructureBuildRangeInfoKHR *ranges =
        gpu_alloc(geometry_count * sizeof(*ranges));
    for (uint32_t i = 0; i < geometry_count; ++i) {
        ranges[i] = (VkAccelerationStructureBuildRangeInfoKHR){
            .primitiveCount = primitive_counts[i],
        };
    }
    const VkAccelerationStructureBuildRangeInfoKHR *range_ptr = ranges;

    VkCommandBuffer cmd = gpu_upload_begin(uploader);
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &range_ptr);
    gpu_upload_end(uploader);

    free(ranges);
    gpu_buffer_destroy(device, &scratch);

    const VkAccelerationStructureDeviceAddressInfoKHR address_info = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .accelerationStructure = accel.handle,
    };
    accel.address = vkGetAccelerationStructureDeviceAddressKHR(device->device, &address_info);

    return accel;
}

gpu_accel_t gpu_blas_build_triangles(gpu_device_t *device, gpu_uploader_t *uploader,
                                     VkDeviceAddress vertices, uint32_t vertex_count,
                                     VkDeviceSize vertex_stride, VkDeviceAddress indices,
                                     uint32_t index_count)
{
    const VkAccelerationStructureGeometryKHR geometry = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
        .geometry.triangles = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
            // Only the position is consumed here; the stride skips the rest of the vertex.
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .vertexData.deviceAddress = vertices,
            .vertexStride = vertex_stride,
            .maxVertex = vertex_count - 1,
            .indexType = VK_INDEX_TYPE_UINT32,
            .indexData.deviceAddress = indices,
        },
    };
    const uint32_t primitive_count = index_count / 3;

    return gpu_accel_build(device, uploader, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                           &geometry, &primitive_count, 1);
}

gpu_accel_t gpu_blas_build_aabbs(gpu_device_t *device, gpu_uploader_t *uploader,
                                 VkDeviceAddress aabbs, uint32_t count)
{
    const VkAccelerationStructureGeometryKHR geometry = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
        // Opaque, so the traversal never invokes an any-hit shader between the intersection
        // shader reporting a hit and the closest hit shader consuming it.
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
        .geometry.aabbs = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
            .data.deviceAddress = aabbs,
            .stride = sizeof(VkAabbPositionsKHR),
        },
    };

    return gpu_accel_build(device, uploader, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                           &geometry, &count, 1);
}

gpu_accel_t gpu_tlas_build(gpu_device_t *device, gpu_uploader_t *uploader,
                           const VkAccelerationStructureInstanceKHR *instances,
                           uint32_t instance_count)
{
    const VkDeviceSize size = instance_count * sizeof(*instances);
    // An empty top level structure is legal and useful -- an editor can hold no entities at
    // all -- but a zero sized buffer is not, so the allocation keeps a floor of one instance.
    gpu_buffer_t instance_buffer = gpu_buffer_create(
        device, size ? size : sizeof(*instances),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        GPU_MEMORY_UPLOAD);
    memcpy(instance_buffer.mapped, instances, (size_t)size);

    const VkAccelerationStructureGeometryKHR geometry = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
        .geometry.instances = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
            .data.deviceAddress = instance_buffer.address,
        },
    };

    gpu_accel_t tlas = gpu_accel_build(device, uploader,
                                       VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &geometry,
                                       &instance_count, 1);

    // gpu_accel_build submits and waits, so the build has consumed the instances by now.
    gpu_buffer_destroy(device, &instance_buffer);
    return tlas;
}

void gpu_accel_destroy(gpu_device_t *device, gpu_accel_t *accel)
{
    if (accel->handle) {
        vkDestroyAccelerationStructureKHR(device->device, accel->handle, NULL);
    }
    gpu_buffer_destroy(device, &accel->buffer);
    memset(accel, 0, sizeof(*accel));
}
