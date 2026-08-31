#include "ui.h"

#include "gpu_internal.h"
#include "scene_io.h"

#include <GLFW/glfw3.h>

#include <string.h>

#define UI_SPIRV_PATH PT_SHADER_DIR "/ui.spv"

// One small window converts to a few thousand vertices at most; these are an order of
// magnitude of headroom over that, and ui_record complains rather than corrupting the draw
// list if they ever do fill up.
#define UI_VERTEX_BUFFER_SIZE (512u * 1024u)
#define UI_INDEX_BUFFER_SIZE (128u * 1024u)

// Nuklear's built-in font is ProggyClean, whose outlines are drawn for a 13 pixel em. Baking
// it at any other size lands every stem on a fractional pixel and the text goes soft.
#define UI_FONT_HEIGHT 13.0f

// Must match the vertex attributes in shaders/ui.slang and the layout below.
typedef struct ui_vertex_t {
    float position[2];
    float uv[2];
    nk_byte color[4];
} ui_vertex_t;

// Matches `struct PushConstants` in shaders/ui.slang.
typedef struct ui_push_constants_t {
    float scale[2];
    float translate[2];
} ui_push_constants_t;

static const struct nk_draw_vertex_layout_element UI_VERTEX_LAYOUT[] = {
    {NK_VERTEX_POSITION, NK_FORMAT_FLOAT, NK_OFFSETOF(ui_vertex_t, position)},
    {NK_VERTEX_TEXCOORD, NK_FORMAT_FLOAT, NK_OFFSETOF(ui_vertex_t, uv)},
    {NK_VERTEX_COLOR, NK_FORMAT_R8G8B8A8, NK_OFFSETOF(ui_vertex_t, color)},
    {NK_VERTEX_LAYOUT_END},
};

// ---------------------------------------------------------------------------
// input
// ---------------------------------------------------------------------------

// Nuklear wants scroll as a delta accumulated since the last frame, and GLFW only delivers
// it through a callback. A file static rather than the window user pointer: main.c already
// keeps its input state this way, and the user pointer stays free for whoever wants it next.
static ui_t *g_scroll_target = NULL;
// Whatever was installed before this module claimed the callback. Chained rather than
// replaced, so the wheel can drive the camera as well -- main.c decides which of the two
// acts on any given event by asking ui_captures_mouse.
static GLFWscrollfun g_previous_scroll = NULL;
static GLFWkeyfun g_previous_key = NULL;
static GLFWcharfun g_previous_char = NULL;

static void on_scroll(GLFWwindow *window, double x, double y)
{
    if (g_scroll_target) {
        g_scroll_target->scroll.x += (float)x;
        g_scroll_target->scroll.y += (float)y;
    }
    if (g_previous_scroll) {
        g_previous_scroll(window, x, y);
    }
}

// Typed characters, which is what actually lands in a text field. Kept separate from the key
// callback because GLFW is the thing that knows the keyboard layout.
//
// Queued rather than pushed straight into nuklear: this runs inside glfwPollEvents, and the
// nk_input_begin that follows would clear it again.
static void on_char(GLFWwindow *window, unsigned int codepoint)
{
    ui_t *ui = g_scroll_target;
    if (ui && ui->text_length < UI_MAX_TEXT_INPUT) {
        ui->text[ui->text_length++] = (nk_rune)codepoint;
    }
    if (g_previous_char) {
        g_previous_char(window, codepoint);
    }
}

// Maps a GLFW key to the nuklear editing key it stands for, or -1 for the great majority
// that mean nothing to a text field -- ordinary characters arrive through on_char instead.
static int nk_key_for(int key, bool control)
{
    if (control) {
        switch (key) {
        case GLFW_KEY_A: return NK_KEY_TEXT_SELECT_ALL;
        case GLFW_KEY_C: return NK_KEY_COPY;
        case GLFW_KEY_V: return NK_KEY_PASTE;
        case GLFW_KEY_X: return NK_KEY_CUT;
        case GLFW_KEY_Z: return NK_KEY_TEXT_UNDO;
        case GLFW_KEY_Y: return NK_KEY_TEXT_REDO;
        default: return -1;
        }
    }

    switch (key) {
    case GLFW_KEY_BACKSPACE: return NK_KEY_BACKSPACE;
    case GLFW_KEY_DELETE: return NK_KEY_DEL;
    case GLFW_KEY_ENTER:
    case GLFW_KEY_KP_ENTER: return NK_KEY_ENTER;
    case GLFW_KEY_TAB: return NK_KEY_TAB;
    case GLFW_KEY_LEFT: return NK_KEY_LEFT;
    case GLFW_KEY_RIGHT: return NK_KEY_RIGHT;
    case GLFW_KEY_UP: return NK_KEY_UP;
    case GLFW_KEY_DOWN: return NK_KEY_DOWN;
    case GLFW_KEY_HOME: return NK_KEY_TEXT_LINE_START;
    case GLFW_KEY_END: return NK_KEY_TEXT_LINE_END;
    default: return -1;
    }
}

// Always chains, so main.c's own bindings keep working.
static void on_key(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    ui_t *ui = g_scroll_target;
    if (ui && ui->key_count < UI_MAX_KEY_EVENTS) {
        const int mapped = nk_key_for(key, (mods & GLFW_MOD_CONTROL) != 0);
        if (mapped >= 0) {
            ui->keys[ui->key_count].key = mapped;
            ui->keys[ui->key_count].down = action != GLFW_RELEASE;
            ++ui->key_count;
        }
    }

    if (g_previous_key) {
        g_previous_key(window, key, scancode, action, mods);
    }
}

// ---------------------------------------------------------------------------
// font atlas
// ---------------------------------------------------------------------------

static void create_font(ui_t *ui)
{
    nk_font_atlas_init_default(&ui->atlas);
    nk_font_atlas_begin(&ui->atlas);

    struct nk_font *font = nk_font_atlas_add_default(&ui->atlas, UI_FONT_HEIGHT, NULL);

    int width = 0;
    int height = 0;
    const void *pixels = nk_font_atlas_bake(&ui->atlas, &width, &height, NK_FONT_ATLAS_ALPHA8);
    if (!pixels || width <= 0 || height <= 0) {
        gpu_fatal("nuklear could not bake its font atlas");
    }

    ui->font = gpu_image_create(ui->device, (VkExtent2D){(uint32_t)width, (uint32_t)height},
                                VK_FORMAT_R8_UNORM,
                                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    gpu_image_upload(&ui->uploader, &ui->font, pixels, (VkDeviceSize)width * (VkDeviceSize)height);

    // This frees the baked pixels, so the upload above has to have finished -- which it has,
    // gpu_image_upload waits on its fence. The texture handle nuklear stamps into every draw
    // command is never read back, since the atlas is the only texture the overlay binds.
    nk_font_atlas_end(&ui->atlas, nk_handle_id(0), &ui->null_texture);

    nk_style_set_font(&ui->context, &font->handle);
}

// ---------------------------------------------------------------------------
// pipeline
// ---------------------------------------------------------------------------

static void create_pipeline(ui_t *ui, VkFormat color_format)
{
    const VkSamplerCreateInfo sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        // Nearest, not linear: the atlas is drawn at 1:1 and never minified, so filtering
        // could only ever interpolate a glyph off its own texels and blur it.
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        // Clamped rather than repeated: a glyph sampled at the edge of its atlas cell must
        // not bleed in the neighbouring one.
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    };
    VK_CHECK(vkCreateSampler(ui->device->device, &sampler_info, NULL, &ui->sampler));

    // A push descriptor set: one texture, rebound every frame, so a pool and a persistent
    // set would both be pure overhead. pushDescriptor is already an enabled device feature.
    const VkDescriptorSetLayoutBinding binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    const VkDescriptorSetLayoutCreateInfo set_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
        .bindingCount = 1,
        .pBindings = &binding,
    };
    VK_CHECK(vkCreateDescriptorSetLayout(ui->device->device, &set_layout_info, NULL,
                                         &ui->set_layout));

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .size = sizeof(ui_push_constants_t),
    };
    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &ui->set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    VK_CHECK(vkCreatePipelineLayout(ui->device->device, &layout_info, NULL, &ui->layout));

    VkShaderModule module = gpu_shader_module_load(ui->device, UI_SPIRV_PATH);
    if (!module) {
        gpu_fatal("could not load the overlay shader from %s", UI_SPIRV_PATH);
    }

    const VkPipelineShaderStageCreateInfo stages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = module,
            .pName = "ui_vertex",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = module,
            .pName = "ui_fragment",
        },
    };

    const VkVertexInputBindingDescription vertex_binding = {
        .binding = 0,
        .stride = sizeof(ui_vertex_t),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    const VkVertexInputAttributeDescription vertex_attributes[] = {
        {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ui_vertex_t, position)},
        {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ui_vertex_t, uv)},
        // UNORM, so the shader receives the packed bytes already scaled into [0,1].
        {2, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(ui_vertex_t, color)},
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
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
    // Both are dynamic; the counts here are all this struct still carries.
    const VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };
    const VkPipelineRasterizationStateCreateInfo rasterization = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        // Nuklear does not promise a winding order, so nothing may be culled.
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisample = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    // Straight "over" compositing. The blend happens in linear space, after the shader has
    // undone nuklear's sRGB encode and before the attachment reapplies it.
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

    // dynamicRendering: the attachment format goes here instead of into a render pass.
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
        .layout = ui->layout,
    };
    const VkResult result = vkCreateGraphicsPipelines(ui->device->device, VK_NULL_HANDLE, 1,
                                                      &pipeline_info, NULL, &ui->pipeline);
    vkDestroyShaderModule(ui->device->device, module, NULL);

    if (result != VK_SUCCESS) {
        gpu_fatal("vkCreateGraphicsPipelines failed for the overlay: %s",
                  vk_result_string(result));
    }
}

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

void ui_init(ui_t *ui, gpu_device_t *device, GLFWwindow *window, VkFormat color_format)
{
    memset(ui, 0, sizeof(*ui));
    ui->device = device;
    ui->window = window;

    gpu_uploader_init(&ui->uploader, device);

    // No font yet: create_font sets it once the atlas has been baked.
    if (!nk_init_default(&ui->context, NULL)) {
        gpu_fatal("could not initialise nuklear");
    }
    nk_buffer_init_default(&ui->commands);

    create_font(ui);
    create_pipeline(ui, color_format);

    ui->convert = (struct nk_convert_config){
        .global_alpha = 1.0f,
        .line_AA = NK_ANTI_ALIASING_ON,
        .shape_AA = NK_ANTI_ALIASING_ON,
        .circle_segment_count = 22,
        .arc_segment_count = 22,
        .curve_segment_count = 22,
        .tex_null = ui->null_texture,
        .vertex_layout = UI_VERTEX_LAYOUT,
        .vertex_size = sizeof(ui_vertex_t),
        .vertex_alignment = NK_ALIGNOF(ui_vertex_t),
    };

    for (uint32_t i = 0; i < GPU_FRAMES_IN_FLIGHT; ++i) {
        ui->geometry[i].vertices = gpu_buffer_create(device, UI_VERTEX_BUFFER_SIZE,
                                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                     GPU_MEMORY_UPLOAD);
        ui->geometry[i].indices = gpu_buffer_create(device, UI_INDEX_BUFFER_SIZE,
                                                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                    GPU_MEMORY_UPLOAD);
    }

    // Mouse position and buttons are polled in ui_begin_frame; only the wheel needs a
    // callback, and it is one main.c does not use, so its key callback stays untouched.
    g_scroll_target = ui;
    g_previous_scroll = glfwSetScrollCallback(window, on_scroll);
    g_previous_key = glfwSetKeyCallback(window, on_key);
    g_previous_char = glfwSetCharCallback(window, on_char);

    snprintf(ui->scene_path, sizeof(ui->scene_path), "%s", PT_SCENE_DIR "/default.pts");
    ui->scene_path_length = (int)strlen(ui->scene_path);
    ui->show_grid = true;
    ui->show_gizmos = true;

    ui->last_time = glfwGetTime();
}

void ui_free(ui_t *ui)
{
    if (!ui->device) {
        return;
    }
    VkDevice device = ui->device->device;

    if (g_scroll_target == ui) {
        glfwSetScrollCallback(ui->window, g_previous_scroll);
        glfwSetKeyCallback(ui->window, g_previous_key);
        glfwSetCharCallback(ui->window, g_previous_char);
        g_previous_scroll = NULL;
        g_previous_key = NULL;
        g_previous_char = NULL;
        g_scroll_target = NULL;
    }

    for (uint32_t i = 0; i < GPU_FRAMES_IN_FLIGHT; ++i) {
        gpu_buffer_destroy(ui->device, &ui->geometry[i].indices);
        gpu_buffer_destroy(ui->device, &ui->geometry[i].vertices);
    }

    vkDestroyPipeline(device, ui->pipeline, NULL);
    vkDestroyPipelineLayout(device, ui->layout, NULL);
    vkDestroyDescriptorSetLayout(device, ui->set_layout, NULL);
    vkDestroySampler(device, ui->sampler, NULL);
    gpu_image_destroy(ui->device, &ui->font);

    gpu_uploader_free(&ui->uploader);

    nk_buffer_free(&ui->commands);
    nk_free(&ui->context);
    // After nk_free: the context's style still points at a font this owns.
    nk_font_atlas_clear(&ui->atlas);

    memset(ui, 0, sizeof(*ui));
}

// ---------------------------------------------------------------------------
// per frame input
// ---------------------------------------------------------------------------

void ui_begin_frame(ui_t *ui)
{
    const double now = glfwGetTime();
    const float delta_ms = (float)((now - ui->last_time) * 1000.0);
    ui->last_time = now;

    // An exponential moving average: settles within a handful of frames, but still reacts
    // fast enough to show what a setting cost the moment it is changed.
    ui->frame_ms = ui->frame_ms > 0.0f ? ui->frame_ms * 0.9f + delta_ms * 0.1f : delta_ms;

    double x = 0.0;
    double y = 0.0;
    glfwGetCursorPos(ui->window, &x, &y);

    struct nk_context *ctx = &ui->context;
    nk_input_begin(ctx);
    nk_input_motion(ctx, (int)x, (int)y);
    // Polled rather than event driven: nuklear derives its own press and release edges from
    // the level, so a callback would buy nothing here.
    nk_input_button(ctx, NK_BUTTON_LEFT, (int)x, (int)y,
                    glfwGetMouseButton(ui->window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    nk_input_button(ctx, NK_BUTTON_MIDDLE, (int)x, (int)y,
                    glfwGetMouseButton(ui->window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    nk_input_button(ctx, NK_BUTTON_RIGHT, (int)x, (int)y,
                    glfwGetMouseButton(ui->window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    nk_input_scroll(ctx, ui->scroll);
    ui->scroll = nk_vec2(0.0f, 0.0f);

    // The whole reason the callbacks queue instead of pushing directly: nk_input_begin above
    // clears both the typed text and every key's click flag, so anything fed during
    // glfwPollEvents would have been thrown away before a widget ever saw it.
    for (uint32_t i = 0; i < ui->key_count; ++i) {
        nk_input_key(ctx, (enum nk_keys)ui->keys[i].key, ui->keys[i].down);
    }
    ui->key_count = 0;

    for (uint32_t i = 0; i < ui->text_length; ++i) {
        nk_input_unicode(ctx, ui->text[i]);
    }
    ui->text_length = 0;

    nk_input_end(ctx);
}

bool ui_captures_mouse(const ui_t *ui)
{
    // The const is dropped because nuklear's queries take a mutable context, though neither
    // of these writes to it.
    struct nk_context *ctx = (struct nk_context *)&ui->context;
    return nk_window_is_any_hovered(ctx) || nk_item_is_any_active(ctx);
}

bool ui_captures_keyboard(const ui_t *ui)
{
    // Set by the edit widgets in ui_draw_overlay when they report NK_EDIT_ACTIVE. Hovering
    // counts too, because a property field is edited with the cursor over the panel.
    return ui->editing_text || ui_captures_mouse(ui);
}

// ---------------------------------------------------------------------------
// the overlay window
// ---------------------------------------------------------------------------

// A labelled float triple on one row, the shape most of the property panel is made of.
// Returns true when the user changed anything, which is the caller's cue to bump the scene
// revision -- nk_property_float reports that itself, but only for the one field.
static bool property_vec3(struct nk_context *ctx, const char *label, float value[3], float min,
                          float max, float step)
{
    nk_layout_row_dynamic(ctx, 16.0f, 1);
    nk_label(ctx, label, NK_TEXT_LEFT);

    float before[3];
    memcpy(before, value, sizeof(before));

    nk_layout_row_dynamic(ctx, 22.0f, 3);
    // The names are hidden behind a # so the three fields fit side by side, but nuklear still
    // needs them distinct: it hashes the name to keep each field's editing state apart.
    nk_property_float(ctx, "#x", min, &value[0], max, step, step * 0.5f);
    nk_property_float(ctx, "#y", min, &value[1], max, step, step * 0.5f);
    nk_property_float(ctx, "#z", min, &value[2], max, step, step * 0.5f);

    return memcmp(before, value, sizeof(before)) != 0;
}

static bool property_float(struct nk_context *ctx, const char *label, float *value, float min,
                           float max, float step)
{
    const float before = *value;
    nk_layout_row_dynamic(ctx, 22.0f, 1);
    nk_property_float(ctx, label, min, value, max, step, step * 0.5f);
    return before != *value;
}

// A swatch that opens a picker, for the values that really are colours. Only for the ones
// that live in [0,1]: emission is HDR and stays a numeric triple, because a 0-to-1 picker
// would quietly clamp away most of what it can express.
static bool property_color(struct nk_context *ctx, const char *label, float rgb[3])
{
    struct nk_colorf color = {rgb[0], rgb[1], rgb[2], 1.0f};

    nk_layout_row_dynamic(ctx, 22.0f, 2);
    nk_label(ctx, label, NK_TEXT_LEFT);
    if (nk_combo_begin_color(ctx, nk_rgb_cf(color), nk_vec2(220.0f, 320.0f))) {
        nk_layout_row_dynamic(ctx, 150.0f, 1);
        color = nk_color_picker(ctx, color, NK_RGB);

        // The numeric fields stay, because a picker is no way to type an exact value.
        nk_layout_row_dynamic(ctx, 22.0f, 1);
        color.r = nk_propertyf(ctx, "#R", 0.0f, color.r, 1.0f, 0.01f, 0.005f);
        color.g = nk_propertyf(ctx, "#G", 0.0f, color.g, 1.0f, 0.01f, 0.005f);
        color.b = nk_propertyf(ctx, "#B", 0.0f, color.b, 1.0f, 0.01f, 0.005f);
        nk_combo_end(ctx);
    }

    if (color.r == rgb[0] && color.g == rgb[1] && color.b == rgb[2]) {
        return false;
    }
    rgb[0] = color.r;
    rgb[1] = color.g;
    rgb[2] = color.b;
    return true;
}

// The list of everything in the scene, plus add and delete. Selection lives here because
// there is no viewport picking.
static void scene_list(struct nk_context *ctx, pt_scene_t *scene, gizmo_selection_t *selection)
{
    nk_layout_row_dynamic(ctx, 22.0f, 2);
    // Tall enough for the analytic shapes plus a few baked models before it has to scroll.
    if (nk_combo_begin_label(ctx, "Add shape", nk_vec2(160.0f, 260.0f))) {
        nk_layout_row_dynamic(ctx, 22.0f, 1);
        for (uint32_t i = 0; i < pt_shape_count(); ++i) {
            if (nk_combo_item_label(ctx, pt_shape_name((pt_shape_t)i), NK_TEXT_LEFT)) {
                if (pt_scene_add_entity(scene, (pt_shape_t)i)) {
                    // Select what was just added: the next thing anyone wants is to place it.
                    selection->target = GIZMO_TARGET_ENTITY;
                    selection->index = scene->entity_count - 1;
                }
            }
        }
        nk_combo_end(ctx);
    }
    if (nk_combo_begin_label(ctx, "Add light", nk_vec2(160.0f, 200.0f))) {
        nk_layout_row_dynamic(ctx, 22.0f, 1);
        for (uint32_t i = 0; i < PT_LIGHT_TYPE_COUNT; ++i) {
            if (nk_combo_item_label(ctx, pt_light_type_name((pt_light_type_t)i), NK_TEXT_LEFT)) {
                if (pt_scene_add_light(scene, (pt_light_type_t)i)) {
                    selection->target = GIZMO_TARGET_LIGHT;
                    selection->index = scene->light_count - 1;
                }
            }
        }
        nk_combo_end(ctx);
    }

    // A group takes the height of the row it is placed in, so this has to be asked for
    // explicitly: without it the list inherits the 22 pixel row above and collapses.
    nk_layout_row_dynamic(ctx, 170.0f, 1);
    if (nk_group_begin(ctx, "scene_items", NK_WINDOW_BORDER)) {
        nk_layout_row_dynamic(ctx, 20.0f, 1);

        for (uint32_t i = 0; i < scene->entity_count; ++i) {
            const bool selected =
                selection->target == GIZMO_TARGET_ENTITY && selection->index == i;
            if (nk_selectable_label(ctx, scene->entities[i].name, NK_TEXT_LEFT,
                                    &(nk_bool){selected}) &&
                !selected) {
                selection->target = GIZMO_TARGET_ENTITY;
                selection->index = i;
            }
        }
        for (uint32_t i = 0; i < scene->light_count; ++i) {
            const bool selected =
                selection->target == GIZMO_TARGET_LIGHT && selection->index == i;
            char label[PT_MAX_NAME + 16];
            snprintf(label, sizeof(label), "* %s", scene->lights[i].name);
            if (nk_selectable_label(ctx, label, NK_TEXT_LEFT, &(nk_bool){selected}) &&
                !selected) {
                selection->target = GIZMO_TARGET_LIGHT;
                selection->index = i;
            }
        }

        nk_group_end(ctx);
    }

    nk_layout_row_dynamic(ctx, 24.0f, 1);
    if (nk_button_label(ctx, "Delete selected")) {
        if (selection->target == GIZMO_TARGET_ENTITY) {
            pt_scene_remove_entity(scene, selection->index);
        } else if (selection->target == GIZMO_TARGET_LIGHT) {
            pt_scene_remove_light(scene, selection->index);
        }
        // The indices behind the removed one all shifted, so the safe move is to select
        // nothing rather than to guess.
        selection->target = GIZMO_TARGET_NONE;
    }
}

static void entity_properties(struct nk_context *ctx, pt_scene_t *scene, uint32_t index,
                              bool *editing)
{
    pt_entity_t *entity = &scene->entities[index];
    bool changed = false;

    nk_layout_row_dynamic(ctx, 22.0f, 1);
    *editing |= (nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, entity->name,
                                                sizeof(entity->name), nk_filter_default) &
                 NK_EDIT_ACTIVE) != 0;

    if (nk_combo_begin_label(ctx, pt_shape_name(entity->shape), nk_vec2(200.0f, 260.0f))) {
        nk_layout_row_dynamic(ctx, 22.0f, 1);
        for (uint32_t i = 0; i < pt_shape_count(); ++i) {
            if (nk_combo_item_label(ctx, pt_shape_name((pt_shape_t)i), NK_TEXT_LEFT) &&
                entity->shape != (pt_shape_t)i) {
                entity->shape = (pt_shape_t)i;
                changed = true;
            }
        }
        nk_combo_end(ctx);
    }

    changed |= property_vec3(ctx, "Position", entity->translation, -1000.0f, 1000.0f, 0.05f);
    changed |= property_vec3(ctx, "Rotation", entity->rotation, -360.0f, 360.0f, 1.0f);
    // Floored well above zero: a zero scale collapses the instance transform and leaves the
    // acceleration structure build with nothing to work with.
    changed |= property_vec3(ctx, "Scale", entity->scale, 0.01f, 1000.0f, 0.05f);
    changed |= property_color(ctx, "Albedo", entity->albedo);
    // Colour and strength apart, exactly as a light carries them: the picker can stay in
    // [0,1] where it belongs while the brightness goes as high as it likes.
    changed |= property_color(ctx, "Emission", entity->emission);
    changed |= property_float(ctx, "Emission strength", &entity->emission_strength, 0.0f,
                              1000.0f, 0.25f);
    changed |= property_float(ctx, "Roughness", &entity->roughness, 0.0f, 1.0f, 0.01f);
    // 0 is a dielectric, 1 a conductor. In between is not physical -- no real surface is
    // half a metal -- but it is the standard authoring control and it is what makes worn or
    // partly-oxidised surfaces easy to dial in.
    changed |= property_float(ctx, "Metallic", &entity->metallic, 0.0f, 1.0f, 0.01f);
    changed |= property_float(ctx, "Transmission", &entity->transmission, 0.0f, 1.0f, 0.01f);
    changed |= property_float(ctx, "IOR", &entity->ior, 1.0f, 3.0f, 0.01f);
    // 0 means no dispersion. Real glasses live between about 20 and 65, so the useful part of
    // this slider is a long way from its left end.
    changed |= property_float(ctx, "Abbe", &entity->abbe, 0.0f, 80.0f, 0.5f);
    // The colour reached after `Absorb dist` of travel through the material. White is inert.
    changed |= property_color(ctx, "Absorption", entity->absorption);
    changed |= property_float(ctx, "Absorb dist", &entity->absorption_distance, 0.01f, 100.0f,
                              0.05f);

    if (changed) {
        ++scene->revision;
    }
}

static void light_properties(struct nk_context *ctx, pt_scene_t *scene, uint32_t index,
                             bool *editing)
{
    pt_light_t *light = &scene->lights[index];
    bool changed = false;

    nk_layout_row_dynamic(ctx, 22.0f, 1);
    *editing |= (nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, light->name,
                                                sizeof(light->name), nk_filter_default) &
                 NK_EDIT_ACTIVE) != 0;

    if (nk_combo_begin_label(ctx, pt_light_type_name(light->type), nk_vec2(200.0f, 200.0f))) {
        nk_layout_row_dynamic(ctx, 22.0f, 1);
        for (uint32_t i = 0; i < PT_LIGHT_TYPE_COUNT; ++i) {
            if (nk_combo_item_label(ctx, pt_light_type_name((pt_light_type_t)i), NK_TEXT_LEFT) &&
                light->type != (pt_light_type_t)i) {
                light->type = (pt_light_type_t)i;
                changed = true;
            }
        }
        nk_combo_end(ctx);
    }

    // Only the fields the chosen type actually reads, so the panel never invites an edit that
    // does nothing.
    if (light->type != PT_LIGHT_DIRECTIONAL) {
        changed |= property_vec3(ctx, "Position", light->position, -1000.0f, 1000.0f, 0.05f);
    }
    if (light->type != PT_LIGHT_POINT) {
        changed |= property_vec3(ctx, "Direction", light->direction, -1.0f, 1.0f, 0.02f);
    }
    // An area light's position is its centre, so it is never the irrelevant field it is for
    // a directional one.
    changed |= property_color(ctx, "Color", light->color);
    changed |= property_float(ctx, "Intensity", &light->intensity, 0.0f, 10000.0f, 1.0f);
    changed |= property_float(ctx, "Range", &light->range, 0.0f, 1000.0f, 0.5f);
    // 0 is the "no temperature" sentinel rather than a real Kelvin value, so the slider stops
    // at 1000 and the step down to 0 is deliberate. A blackbody changes only the hue -- the
    // normalisation in fill_light holds the luminance -- so this can be swept without the
    // exposure moving under you.
    changed |= property_float(ctx, "Temperature K", &light->temperature, 0.0f, 12000.0f,
                              50.0f);
    if (light->type == PT_LIGHT_AREA) {
        nk_layout_row_dynamic(ctx, 16.0f, 1);
        nk_label(ctx, "Size (half extents)", NK_TEXT_LEFT);
        const float before[2] = {light->size[0], light->size[1]};
        nk_layout_row_dynamic(ctx, 22.0f, 2);
        nk_property_float(ctx, "#u", 0.01f, &light->size[0], 100.0f, 0.05f, 0.02f);
        nk_property_float(ctx, "#v", 0.01f, &light->size[1], 100.0f, 0.05f, 0.02f);
        changed |= before[0] != light->size[0] || before[1] != light->size[1];
    }
    if (light->type == PT_LIGHT_SPOT) {
        changed |= property_float(ctx, "Cone inner", &light->cone_inner, 0.0f, 89.0f, 1.0f);
        changed |= property_float(ctx, "Cone outer", &light->cone_outer, 0.0f, 89.0f, 1.0f);
        // The falloff divides by their difference, so the inner cone can never overtake the
        // outer one.
        if (light->cone_inner > light->cone_outer) {
            light->cone_inner = light->cone_outer;
        }
    }

    if (changed) {
        ++scene->revision;
    }
}

// Works around a regression in nuklear 4.13.0 ("Fix: nk_property not updating
// 'win->edit.active'" in its own changelog). When a number field starts being edited,
// nk_property now sets win->edit.active -- but it never touches win->edit.name, which still
// holds the identity of whichever text field was last focused in this window. nk_edit_buffer
// decides whether a text field is the focused one with
//
//     hash = win->edit.seq++;
//     if (win->edit.active && hash == win->edit.name) ...
//
// so that text field quietly reactivates itself and consumes the very keystrokes being typed
// into the number: the digits land in the entity's name as well, and neither field ends up
// holding what was typed. win->edit.name is zero until something is focused, which is why even
// a fresh session sees it -- the first text field in the window hashes to zero too.
//
// A property and a text field can never both legitimately be editing, since they share the one
// ctx->text_edit, so a live property means no text field owns the keyboard. Saying that before
// the window lays out any edit widget is the whole fix; nk_property sets the flag again on the
// next frame it wants it. Nothing else reads it -- nk_item_is_any_active goes by hover and the
// last widget state -- so the camera still stays put while a number is being typed.
static void clear_property_edit_collision(struct nk_context *ctx)
{
    struct nk_window *win = ctx->current;
    if (win && win->property.active) {
        win->edit.active = nk_false;
    }
}

static void scene_window(ui_t *ui, renderer_t *renderer, gizmo_selection_t *selection,
                         gizmo_t *gizmo)
{
    struct nk_context *ctx = &ui->context;
    pt_scene_t *scene = &renderer->scene;

    if (nk_begin(ctx, "Scene", nk_rect(20.0f, 300.0f, 300.0f, 620.0f),
                 NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE |
                     NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE)) {
        // This window is the one that mixes text fields with number fields, so it is the one
        // that can hit the collision above. Has to come before the first edit widget.
        clear_property_edit_collision(ctx);

        nk_layout_row_dynamic(ctx, 22.0f, 1);
        ui->editing_text |= (nk_edit_string(ctx, NK_EDIT_FIELD, ui->scene_path,
                                            &ui->scene_path_length, UI_PATH_MAX - 1,
                                            nk_filter_default) &
                             NK_EDIT_ACTIVE) != 0;
        ui->scene_path[ui->scene_path_length] = '\0';

        nk_layout_row_dynamic(ctx, 24.0f, 2);
        if (nk_button_label(ctx, "Save")) {
            pt_scene_save(scene, ui->scene_path);
        }
        if (nk_button_label(ctx, "Load")) {
            // Indices from the old scene mean nothing in the new one.
            selection->target = GIZMO_TARGET_NONE;
            pt_scene_load(scene, ui->scene_path);
        }

        // The gizmo mode. Also bound to 1/2/3 in main.c, and both write the same field.
        nk_layout_row_dynamic(ctx, 22.0f, 3);
        static const char *const MODE_NAMES[GIZMO_MODE_COUNT] = {"Move", "Rotate", "Scale"};
        for (uint32_t i = 0; i < GIZMO_MODE_COUNT; ++i) {
            if (nk_option_label(ctx, MODE_NAMES[i], gizmo->mode == (gizmo_mode_t)i)) {
                gizmo->mode = (gizmo_mode_t)i;
            }
        }

        scene_list(ctx, scene, selection);

        nk_layout_row_dynamic(ctx, 16.0f, 1);
        if (selection->target == GIZMO_TARGET_ENTITY && selection->index < scene->entity_count) {
            entity_properties(ctx, scene, selection->index, &ui->editing_text);
        } else if (selection->target == GIZMO_TARGET_LIGHT &&
                   selection->index < scene->light_count) {
            light_properties(ctx, scene, selection->index, &ui->editing_text);
        } else {
            nk_label(ctx, "Nothing selected", NK_TEXT_LEFT);
        }
    }
    nk_end(ctx);
}

void ui_draw_overlay(ui_t *ui, renderer_t *renderer, gizmo_selection_t *selection,
                     gizmo_t *gizmo)
{
    struct nk_context *ctx = &ui->context;

    // Recomputed from scratch each frame by the edit widgets below.
    ui->editing_text = false;

    scene_window(ui, renderer, selection, gizmo);

    // Tall enough for every row below: the panel has no scrollbar, so anything that does not
    // fit is simply unreachable.
    if (nk_begin(ctx, "Pathtracer", nk_rect(20.0f, 20.0f, 280.0f, 640.0f),
                 NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_MINIMIZABLE |
                     NK_WINDOW_TITLE | NK_WINDOW_NO_SCROLLBAR)) {
        const float fps = ui->frame_ms > 0.0f ? 1000.0f / ui->frame_ms : 0.0f;

        nk_layout_row_dynamic(ctx, 16.0f, 2);
        nk_label(ctx, "Frame time", NK_TEXT_LEFT);
        nk_labelf(ctx, NK_TEXT_RIGHT, "%.2f ms", (double)ui->frame_ms);
        nk_label(ctx, "Rate", NK_TEXT_LEFT);
        nk_labelf(ctx, NK_TEXT_RIGHT, "%.0f fps", (double)fps);
        nk_label(ctx, "Frame index", NK_TEXT_LEFT);
        nk_labelf(ctx, NK_TEXT_RIGHT, "%u", renderer->accum_frames);
        nk_label(ctx, "Samples/px", NK_TEXT_LEFT);
        nk_labelf(ctx, NK_TEXT_RIGHT, "%u",
                  renderer->accum_frames * renderer->settings.samples_per_frame);

        nk_layout_row_dynamic(ctx, 26.0f, 1);
        if (nk_button_label(ctx, "Reset history")) {
            renderer_reset_accumulation(renderer);
        }

        // Nuklear's widgets work on int, while pt_settings_t is all uint32_t so that the
        // memcmp in renderer_record has no padding to trip over. Hence the copy out and
        // back rather than casting the members in place.
        int unlit = renderer->settings.unlit != 0;
        int bounces = (int)renderer->settings.max_bounces;
        int samples = (int)renderer->settings.samples_per_frame;

        // Purely view toggles: they change nothing the tracer does, so unlike the settings
        // below they must not restart the accumulation.
        int grid = ui->show_grid;
        int gizmos = ui->show_gizmos;
        nk_layout_row_dynamic(ctx, 22.0f, 3);
        nk_checkbox_label(ctx, "Unlit", &unlit);
        nk_checkbox_label(ctx, "Grid", &grid);
        nk_checkbox_label(ctx, "Gizmos", &gizmos);
        ui->show_grid = grid != 0;
        ui->show_gizmos = gizmos != 0;

        nk_layout_row_dynamic(ctx, 22.0f, 1);
        nk_property_int(ctx, "Bounces", 1, &bounces, 32, 1, 0.05f);
        nk_property_int(ctx, "Samples/frame", 1, &samples, 32, 1, 0.05f);
        // 0 leaves only the explicit lights, which is the honest way to judge a lighting rig.
        nk_property_float(ctx, "Sky", 0.0f, &renderer->settings.sky_intensity, 4.0f, 0.05f,
                          0.02f);

        // The Preetham sky's own parameters. Turbidity is floored at 1.7 because the model is
        // a fit and goes visibly wrong below that -- the fit had no data there.
        nk_property_float(ctx, "Turbidity", 1.7f, &renderer->settings.turbidity, 10.0f, 0.1f,
                          0.05f);
        // Below the horizon the sky is replaced by a stand-in ground, so negative elevations
        // are allowed but only interesting near zero.
        nk_property_float(ctx, "Sun elevation", -5.0f, &renderer->settings.sun_elevation, 90.0f,
                          1.0f, 0.5f);
        nk_property_float(ctx, "Sun azimuth", 0.0f, &renderer->settings.sun_azimuth, 360.0f,
                          1.0f, 0.5f);
        // 0.53 is the real sun. Larger softens the shadow edges without changing how much
        // light arrives, because sun_radiance normalises by the disc's solid angle.
        nk_property_float(ctx, "Sun size", 0.1f, &renderer->settings.sun_angular_diameter,
                          20.0f, 0.1f, 0.05f);

        renderer->settings.unlit = unlit ? 1u : 0u;
        renderer->settings.max_bounces = (uint32_t)bounces;
        renderer->settings.samples_per_frame = (uint32_t)samples;

        // The a-trous filter. A post-process over the already averaged image, so like the
        // view toggles above -- and unlike everything between them -- it lives outside
        // pt_settings_t and never restarts the accumulation. Reach for it when glass or
        // dispersion has left grain the estimator is converging out only slowly.
        nk_layout_row_dynamic(ctx, 8.0f, 1);
        nk_spacing(ctx, 1);
        nk_layout_row_dynamic(ctx, 22.0f, 1);
        nk_label(ctx, "Denoise", NK_TEXT_LEFT);

        denoise_settings_t *denoise = &renderer->denoise.settings;
        int denoise_on = denoise->enabled;
        int iterations = (int)denoise->iterations;
        nk_checkbox_label(ctx, "Enabled", &denoise_on);
        // Each pass doubles the tap spacing, so five reaches a 65 pixel support. Past that
        // the widest taps are further apart than most of what is on screen.
        nk_property_int(ctx, "Iterations", 1, &iterations, 5, 1, 0.05f);
        // How far two pixels may differ in brightness before they stop filtering each other.
        // Small keeps detail and noise alike; large flattens both.
        nk_property_float(ctx, "Colour phi", 0.01f, &denoise->phi_color, 4.0f, 0.05f, 0.01f);
        // An exponent on the cosine between normals, so it is only interesting at this scale.
        nk_property_float(ctx, "Normal phi", 1.0f, &denoise->phi_normal, 128.0f, 1.0f, 0.5f);
        // World units, and scaled by the tap spacing inside the shader.
        nk_property_float(ctx, "Depth phi", 0.001f, &denoise->phi_depth, 1.0f, 0.005f, 0.002f);
        denoise->enabled = denoise_on != 0;
        denoise->iterations = (uint32_t)iterations;

        // Tonemapping. A post-process like the denoiser above, and outside pt_settings_t for
        // the same reason: it shapes the averaged image rather than changing what is averaged.
        nk_layout_row_dynamic(ctx, 8.0f, 1);
        nk_spacing(ctx, 1);
        nk_layout_row_dynamic(ctx, 22.0f, 1);
        nk_label(ctx, "Tonemap", NK_TEXT_LEFT);

        tonemap_settings_t *tonemap = &renderer->tonemap.settings;
        int tonemap_on = tonemap->enabled;
        nk_checkbox_label(ctx, "Enabled", &tonemap_on);
        tonemap->enabled = tonemap_on != 0;

        // "None" is the naked clip the renderer applied before this pass existed, kept so the
        // spectral work can still be judged against an image nothing has shaped.
        const int picked = nk_combo(ctx, TONEMAP_CURVE_NAMES, (int)TONEMAP_CURVE_COUNT,
                                    (int)tonemap->curve, 22, nk_vec2(nk_widget_width(ctx), 120.0f));
        tonemap->curve = (tonemap_curve_t)picked;

        // Stops, so 0 leaves the scene at the brightness it was authored for.
        nk_property_float(ctx, "Exposure", -8.0f, &tonemap->exposure, 8.0f, 0.1f, 0.05f);

        // The lens. Unlike the two blocks above these *are* scene data -- they save into the
        // .pts alongside the field of view, and changing either restarts the accumulation by
        // itself, because pt_camera_t is part of what renderer_record compares.
        nk_layout_row_dynamic(ctx, 8.0f, 1);
        nk_spacing(ctx, 1);
        nk_layout_row_dynamic(ctx, 22.0f, 1);
        nk_label(ctx, "Camera", NK_TEXT_LEFT);

        // 0 is a pinhole, so the whole effect switches off at the bottom of the range rather
        // than needing a checkbox of its own.
        nk_property_float(ctx, "Aperture", 0.0f, &renderer->scene.camera_aperture, 1.0f, 0.005f,
                          0.002f);
        nk_property_float(ctx, "Focus dist", 0.1f, &renderer->scene.camera_focus_distance,
                          100.0f, 0.1f, 0.05f);

        // Pulling focus by hand means guessing a distance and watching what sharpens; this
        // reads it off the selection instead. Straight-line distance rather than distance
        // along the view axis: the thing being focused on is normally what is being looked
        // at, and where it is not, the straight line is the more useful of the two.
        nk_layout_row_dynamic(ctx, 26.0f, 1);
        if (nk_button_label(ctx, "Focus on selection")) {
            const float *target = NULL;
            if (selection->target == GIZMO_TARGET_ENTITY &&
                selection->index < renderer->scene.entity_count) {
                target = renderer->scene.entities[selection->index].translation;
            } else if (selection->target == GIZMO_TARGET_LIGHT &&
                       selection->index < renderer->scene.light_count) {
                target = renderer->scene.lights[selection->index].position;
            }
            if (target) {
                renderer->scene.camera_focus_distance =
                    pt_vec3_distance(renderer->scene.camera_position, target);
            }
        }
    }
    nk_end(ctx);
}

// ---------------------------------------------------------------------------
// recording
// ---------------------------------------------------------------------------

// Nuklear emits clip rectangles in logical units, and uses a huge sentinel rectangle to mean
// "no clipping", so this both rescales into framebuffer pixels and clamps: Vulkan rejects a
// scissor that is negative or reaches outside the render area.
static VkRect2D clip_to_scissor(struct nk_rect clip, float scale_x, float scale_y,
                                VkExtent2D extent)
{
    float left = clip.x * scale_x;
    float top = clip.y * scale_y;
    float right = (clip.x + clip.w) * scale_x;
    float bottom = (clip.y + clip.h) * scale_y;

    left = left < 0.0f ? 0.0f : left;
    top = top < 0.0f ? 0.0f : top;
    right = right > (float)extent.width ? (float)extent.width : right;
    bottom = bottom > (float)extent.height ? (float)extent.height : bottom;

    // A rectangle entirely off screen ends up inverted after the clamps above.
    right = right < left ? left : right;
    bottom = bottom < top ? top : bottom;

    return (VkRect2D){
        .offset = {(int32_t)left, (int32_t)top},
        .extent = {(uint32_t)(right - left), (uint32_t)(bottom - top)},
    };
}

static void record_draws(ui_t *ui, gpu_frame_t *frame, float scale_x, float scale_y,
                         const ui_push_constants_t *push)
{
    VkCommandBuffer cmd = frame->cmd;
    const ui_geometry_t *geometry = &ui->geometry[frame->slot];

    // The blit in renderer_record filled this image; the overlay now draws on top of it,
    // which is why the attachment loads rather than clears below.
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

    gpu_cmd_begin_rendering(cmd, frame->view, frame->extent, NULL);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ui->pipeline);

    const VkDescriptorImageInfo font_info = {
        .sampler = ui->sampler,
        .imageView = ui->font.view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkWriteDescriptorSet font_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &font_info,
    };
    vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ui->layout, 0, 1, &font_write);

    const VkViewport viewport = {
        .width = (float)frame->extent.width,
        .height = (float)frame->extent.height,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdPushConstants(cmd, ui->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(*push), push);

    const VkDeviceSize vertex_offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &geometry->vertices.handle, &vertex_offset);
    vkCmdBindIndexBuffer(cmd, geometry->indices.handle, 0, VK_INDEX_TYPE_UINT32);

    // The draw commands index one shared, already ordered element buffer, so the offset just
    // walks forward; only the scissor changes between them.
    uint32_t first_index = 0;
    const struct nk_draw_command *draw = NULL;
    nk_draw_foreach(draw, &ui->context, &ui->commands)
    {
        if (draw->elem_count == 0) {
            continue;
        }
        const VkRect2D scissor = clip_to_scissor(draw->clip_rect, scale_x, scale_y,
                                                 frame->extent);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdDrawIndexed(cmd, draw->elem_count, 1, first_index, 0, 0);
        first_index += draw->elem_count;
    }

    gpu_cmd_end_rendering(cmd);

    // No barrier afterwards: gpu_frame_end's transition to PRESENT_SRC_KHR sources from
    // ALL_COMMANDS, which already covers these colour writes.
}

void ui_record(ui_t *ui, gpu_frame_t *frame)
{
    ui_geometry_t *geometry = &ui->geometry[frame->slot];

    struct nk_buffer vertices;
    struct nk_buffer indices;
    nk_buffer_init_fixed(&vertices, geometry->vertices.mapped,
                         (nk_size)geometry->vertices.size);
    nk_buffer_init_fixed(&indices, geometry->indices.mapped, (nk_size)geometry->indices.size);

    const nk_flags converted = nk_convert(&ui->context, &ui->commands, &vertices, &indices,
                                          &ui->convert);

    // Nuklear works in logical window units while the swapchain is in framebuffer pixels.
    // Handing the shader the logical size and stretching the viewport over the whole
    // framebuffer is what makes the overlay hold its physical size on a HiDPI display.
    int window_width = 0;
    int window_height = 0;
    glfwGetWindowSize(ui->window, &window_width, &window_height);

    if (converted != NK_CONVERT_SUCCESS) {
        // A partial conversion leaves draw commands pointing at elements that were never
        // written, so the whole frame is dropped rather than drawn from a torn buffer.
        fprintf(stderr, "ui: nk_convert ran out of room (0x%x), skipping the overlay\n",
                converted);
    } else if (window_width > 0 && window_height > 0) {
        const ui_push_constants_t push = {
            .scale = {2.0f / (float)window_width, 2.0f / (float)window_height},
            .translate = {-1.0f, -1.0f},
        };
        record_draws(ui, frame, (float)frame->extent.width / (float)window_width,
                     (float)frame->extent.height / (float)window_height, &push);
    }

    // Ends the nuklear frame whatever happened above; skipping this corrupts the next one.
    nk_clear(&ui->context);
    nk_buffer_clear(&ui->commands);
}
