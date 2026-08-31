#include "debug_draw.h"

#include "gpu_internal.h"

#include <math.h>
#include <string.h>

#define DEBUG_SPIRV_PATH PT_SHADER_DIR "/debug.spv"

// Matches `struct PushConstants` in shaders/debug.slang: the view projection as four rows,
// and whether this batch may be hidden by the scene.
typedef struct debug_push_constants_t {
    float rows[4][4];
    uint32_t depth_test;
} debug_push_constants_t;

// The vertex stage transforms with the rows, the fragment stage tests against depth_test.
// vkCmdPushConstants has to name exactly the range's stages, so both sides take it from here.
#define DEBUG_PUSH_CONSTANT_STAGES (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)

// ---------------------------------------------------------------------------
// pipeline
// ---------------------------------------------------------------------------

static void create_pipeline(debug_draw_t *debug, VkFormat color_format)
{
    // The tracer's depth image, pushed inline: it is rebound every frame and is the only
    // descriptor this pass has, so a pool would be pure overhead.
    const VkDescriptorSetLayoutBinding binding = {
        .binding = 0,
        // Sampled rather than storage: a fragment shader may only hold a writable storage
        // image when fragmentStoresAndAtomics is enabled, and this one only ever reads.
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    const VkDescriptorSetLayoutCreateInfo set_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
        .bindingCount = 1,
        .pBindings = &binding,
    };
    VK_CHECK(vkCreateDescriptorSetLayout(debug->device->device, &set_layout_info, NULL,
                                         &debug->set_layout));

    const VkPushConstantRange push_range = {
        .stageFlags = DEBUG_PUSH_CONSTANT_STAGES,
        .size = sizeof(debug_push_constants_t),
    };
    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &debug->set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    VK_CHECK(vkCreatePipelineLayout(debug->device->device, &layout_info, NULL, &debug->layout));

    VkShaderModule module = gpu_shader_module_load(debug->device, DEBUG_SPIRV_PATH);
    if (!module) {
        gpu_fatal("could not load the debug line shader from %s", DEBUG_SPIRV_PATH);
    }

    const VkPipelineShaderStageCreateInfo stages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = module,
            .pName = "debug_vertex",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = module,
            .pName = "debug_fragment",
        },
    };

    const VkVertexInputBindingDescription vertex_binding = {
        .binding = 0,
        .stride = sizeof(debug_vertex_t),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    const VkVertexInputAttributeDescription vertex_attributes[] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(debug_vertex_t, position)},
        {1, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(debug_vertex_t, color)},
    };
    const VkPipelineVertexInputStateCreateInfo vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertex_binding,
        .vertexAttributeDescriptionCount = PT_COUNT(vertex_attributes),
        .pVertexAttributeDescriptions = vertex_attributes,
    };

    const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
    };
    const VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };
    const VkPipelineRasterizationStateCreateInfo rasterization = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        // wideLines is not an enabled device feature, so this is the only legal value.
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisample = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    // Blended so a grid can sit under the scene without drowning it.
    const VkPipelineColorBlendAttachmentState blend_attachment = {
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo blend = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &blend_attachment,
    };

    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = PT_COUNT(dynamic_states),
        .pDynamicStates = dynamic_states,
    };

    const VkPipelineRenderingCreateInfo rendering = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &color_format,
    };
    const VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rendering,
        .stageCount = PT_COUNT(stages),
        .pStages = stages,
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pColorBlendState = &blend,
        .pDynamicState = &dynamic,
        .layout = debug->layout,
    };
    const VkResult result = vkCreateGraphicsPipelines(debug->device->device, VK_NULL_HANDLE, 1,
                                                      &pipeline_info, NULL, &debug->pipeline);
    vkDestroyShaderModule(debug->device->device, module, NULL);

    if (result != VK_SUCCESS) {
        gpu_fatal("vkCreateGraphicsPipelines failed for the debug lines: %s",
                  vk_result_string(result));
    }
}

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

void debug_draw_init(debug_draw_t *debug, gpu_device_t *device, VkFormat color_format)
{
    memset(debug, 0, sizeof(*debug));
    debug->device = device;

    create_pipeline(debug, color_format);

    // Both lists are concatenated into the one buffer, so it holds the sum of their caps.
    for (uint32_t i = 0; i < GPU_FRAMES_IN_FLIGHT; ++i) {
        debug->vertices[i] = gpu_buffer_create(
            device, sizeof(debug->occluded) + sizeof(debug->overlay),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, GPU_MEMORY_UPLOAD);
    }
}

void debug_draw_free(debug_draw_t *debug)
{
    if (!debug->device) {
        return;
    }
    VkDevice device = debug->device->device;

    for (uint32_t i = 0; i < GPU_FRAMES_IN_FLIGHT; ++i) {
        gpu_buffer_destroy(debug->device, &debug->vertices[i]);
    }
    vkDestroyPipeline(device, debug->pipeline, NULL);
    vkDestroyPipelineLayout(device, debug->layout, NULL);
    vkDestroyDescriptorSetLayout(device, debug->set_layout, NULL);

    memset(debug, 0, sizeof(*debug));
}

// ---------------------------------------------------------------------------
// queueing
// ---------------------------------------------------------------------------

void debug_draw_begin(debug_draw_t *debug)
{
    debug->occluded_count = 0;
    debug->overlay_count = 0;
    debug->overlay_mode = false;
}

void debug_draw_set_overlay(debug_draw_t *debug, bool overlay)
{
    debug->overlay_mode = overlay;
}

void debug_draw_line(debug_draw_t *debug, const float a[3], const float b[3], uint32_t rgba)
{
    debug_vertex_t *list = debug->overlay_mode ? debug->overlay : debug->occluded;
    uint32_t *count = debug->overlay_mode ? &debug->overlay_count : &debug->occluded_count;

    if (*count + 2 > DEBUG_DRAW_MAX_VERTICES) {
        if (!debug->overflowed) {
            debug->overflowed = true;
            fprintf(stderr, "debug_draw: line budget (%u vertices per list) exhausted\n",
                    DEBUG_DRAW_MAX_VERTICES);
        }
        return;
    }

    const uint8_t color[4] = {
        (uint8_t)(rgba >> 24), (uint8_t)(rgba >> 16), (uint8_t)(rgba >> 8), (uint8_t)rgba,
    };

    debug_vertex_t *out = &list[*count];
    memcpy(out[0].position, a, sizeof(out[0].position));
    memcpy(out[0].color, color, sizeof(color));
    memcpy(out[1].position, b, sizeof(out[1].position));
    memcpy(out[1].color, color, sizeof(color));
    *count += 2;
}

void debug_draw_box(debug_draw_t *debug, const VkTransformMatrixKHR *transform, uint32_t rgba)
{
    // The eight corners of the canonical [-1,1] cube every shape is defined inside, put
    // through the instance's own transform so the box follows its rotation and scale.
    static const float CORNERS[8][3] = {
        {-1.0f, -1.0f, -1.0f}, {1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, -1.0f}, {-1.0f, 1.0f, -1.0f},
        {-1.0f, -1.0f, 1.0f},  {1.0f, -1.0f, 1.0f},  {1.0f, 1.0f, 1.0f},  {-1.0f, 1.0f, 1.0f},
    };
    static const uint32_t EDGES[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, // back face
        {4, 5}, {5, 6}, {6, 7}, {7, 4}, // front face
        {0, 4}, {1, 5}, {2, 6}, {3, 7}, // the struts between them
    };

    float world[8][3];
    for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t row = 0; row < 3; ++row) {
            world[i][row] = transform->matrix[row][0] * CORNERS[i][0] +
                            transform->matrix[row][1] * CORNERS[i][1] +
                            transform->matrix[row][2] * CORNERS[i][2] +
                            transform->matrix[row][3];
        }
    }

    for (uint32_t i = 0; i < 12; ++i) {
        debug_draw_line(debug, world[EDGES[i][0]], world[EDGES[i][1]], rgba);
    }
}

void debug_draw_sphere(debug_draw_t *debug, const float center[3], float radius, uint32_t rgba)
{
    #define DEBUG_RING_SEGMENTS 24

    // Three axis aligned rings. Not a real wireframe, but it reads unambiguously as a sphere
    // and costs a fraction of the lines one would.
    for (uint32_t axis = 0; axis < 3; ++axis) {
        const uint32_t u = (axis + 1) % 3;
        const uint32_t v = (axis + 2) % 3;

        float previous[3];
        for (uint32_t i = 0; i <= DEBUG_RING_SEGMENTS; ++i) {
            const float angle = (float)i / DEBUG_RING_SEGMENTS * 6.28318530718f;

            float point[3];
            memcpy(point, center, sizeof(point));
            point[u] += cosf(angle) * radius;
            point[v] += sinf(angle) * radius;

            if (i > 0) {
                debug_draw_line(debug, previous, point, rgba);
            }
            memcpy(previous, point, sizeof(previous));
        }
    }

    #undef DEBUG_RING_SEGMENTS
}

void debug_draw_grid(debug_draw_t *debug, float extent, float step, uint32_t rgba)
{
    if (step <= 0.0f) {
        return;
    }

    for (float offset = -extent; offset <= extent + 1e-4f; offset += step) {
        const float along_x[2][3] = {{-extent, 0.0f, offset}, {extent, 0.0f, offset}};
        const float along_z[2][3] = {{offset, 0.0f, -extent}, {offset, 0.0f, extent}};
        debug_draw_line(debug, along_x[0], along_x[1], rgba);
        debug_draw_line(debug, along_z[0], along_z[1], rgba);
    }
}

// ---------------------------------------------------------------------------
// recording
// ---------------------------------------------------------------------------

void debug_draw_record(debug_draw_t *debug, gpu_frame_t *frame, const pt_camera_t *camera,
                       const gpu_image_t *depth)
{
    const uint32_t total = debug->occluded_count + debug->overlay_count;
    if (total == 0) {
        return;
    }

    VkCommandBuffer cmd = frame->cmd;
    const gpu_buffer_t *vertices = &debug->vertices[frame->slot];

    // Concatenated, occluded first, so the two draws below are just offsets into one buffer.
    debug_vertex_t *mapped = vertices->mapped;
    memcpy(mapped, debug->occluded, debug->occluded_count * sizeof(debug_vertex_t));
    memcpy(mapped + debug->occluded_count, debug->overlay,
           debug->overlay_count * sizeof(debug_vertex_t));

    // The blit in renderer_record filled this image, and raygen wrote the depth image; the
    // lines both draw over the one and read the other.
    gpu_cmd_image_barrier(cmd, &(gpu_image_barrier_t){
        .image = frame->image,
        .old_layout = VK_IMAGE_LAYOUT_GENERAL,
        .new_layout = VK_IMAGE_LAYOUT_GENERAL,
        .src_stage = VK_PIPELINE_STAGE_2_BLIT_BIT,
        .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dst_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
    });
    gpu_cmd_image_barrier(cmd, &(gpu_image_barrier_t){
        .image = depth->handle,
        .old_layout = VK_IMAGE_LAYOUT_GENERAL,
        .new_layout = VK_IMAGE_LAYOUT_GENERAL,
        .src_stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        .src_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dst_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dst_access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
    });

    gpu_cmd_begin_rendering(cmd, frame->view, frame->extent, NULL);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, debug->pipeline);

    const VkDescriptorImageInfo depth_info = {
        .imageView = depth->view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkWriteDescriptorSet depth_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .pImageInfo = &depth_info,
    };
    vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, debug->layout, 0, 1,
                           &depth_write);

    const VkViewport viewport = {
        .width = (float)frame->extent.width,
        .height = (float)frame->extent.height,
        .maxDepth = 1.0f,
    };
    const VkRect2D scissor = {.extent = frame->extent};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    const float aspect = (float)frame->extent.width / (float)frame->extent.height;
    const pt_mat4_t view_projection = pt_camera_view_projection(camera, aspect);
    debug_push_constants_t push;
    memcpy(push.rows, view_projection.m, sizeof(push.rows));

    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertices->handle, &offset);

    // Two draws over one buffer, differing only in whether the fragment shader consults the
    // scene depth. The overlay batch goes second so it lands on top of the occluded one.
    if (debug->occluded_count > 0) {
        push.depth_test = 1;
        vkCmdPushConstants(cmd, debug->layout, DEBUG_PUSH_CONSTANT_STAGES, 0, sizeof(push),
                           &push);
        vkCmdDraw(cmd, debug->occluded_count, 1, 0, 0);
    }
    if (debug->overlay_count > 0) {
        push.depth_test = 0;
        vkCmdPushConstants(cmd, debug->layout, DEBUG_PUSH_CONSTANT_STAGES, 0, sizeof(push),
                           &push);
        vkCmdDraw(cmd, debug->overlay_count, 1, debug->occluded_count, 0);
    }

    gpu_cmd_end_rendering(cmd);
}
