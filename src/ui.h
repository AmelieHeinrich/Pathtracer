#pragma once

#include "gizmo.h"
#include "gpu.h"
#include "nuklear_config.h"
#include "renderer.h"

// Long enough for a path a person would actually type by hand.
#define UI_PATH_MAX 256

// A frame's worth of typing. Anything beyond this in a single frame is dropped, which takes
// a genuinely implausible burst.
#define UI_MAX_TEXT_INPUT 64
#define UI_MAX_KEY_EVENTS 64

typedef struct ui_key_event_t {
    int key; // an NK_KEY_* value
    bool down;
} ui_key_event_t;

// Geometry for one frame in flight. Nuklear reconverts its whole draw list every frame, so
// these are host visible and written straight through their persistent mapping. One pair per
// slot: frame N must not overwrite the geometry frame N-1 is still drawing from.
typedef struct ui_geometry_t {
    gpu_buffer_t vertices;
    gpu_buffer_t indices;
} ui_geometry_t;

typedef struct ui_t {
    gpu_device_t *device; // not owned
    GLFWwindow *window;   // not owned
    gpu_uploader_t uploader;

    struct nk_context context;
    struct nk_font_atlas atlas;
    struct nk_buffer commands; // draw commands nk_convert emits alongside the geometry
    struct nk_draw_null_texture null_texture;
    struct nk_convert_config convert;

    gpu_image_t font;
    VkSampler sampler;

    // The atlas is the only texture the overlay ever binds, so it is pushed inline rather
    // than allocated from a pool.
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkPipeline pipeline;

    ui_geometry_t geometry[GPU_FRAMES_IN_FLIGHT];

    struct nk_vec2 scroll; // accumulated by the GLFW callback, drained once per frame

    // GLFW delivers input during glfwPollEvents, which runs before nk_input_begin -- and
    // nk_input_begin *clears* the typed text and the key click flags. So keyboard events are
    // queued here by the callbacks and replayed inside the begin/end pair instead of being
    // handed to nuklear as they arrive, where they would simply be wiped.
    nk_rune text[UI_MAX_TEXT_INPUT];
    uint32_t text_length;
    ui_key_event_t keys[UI_MAX_KEY_EVENTS];
    uint32_t key_count;
    // Latched while a text field holds focus, so the camera does not fly off on WASD while
    // someone is typing a filename.
    bool editing_text;

    // View toggles rather than render settings: they change nothing the tracer does, so they
    // live here and never restart the accumulation. Read by main.c when it draws the scene.
    bool show_grid;
    bool show_gizmos;

    double last_time;
    float frame_ms; // smoothed, because a raw per-frame delta is unreadable

    // The scene file the Save and Load buttons act on, editable in the panel. nuklear's text
    // widget wants the length alongside the buffer rather than a terminator.
    char scene_path[UI_PATH_MAX];
    int scene_path_length;
} ui_t;

// `color_format` is the swapchain format the overlay renders into.
void ui_init(ui_t *ui, gpu_device_t *device, GLFWwindow *window, VkFormat color_format);
void ui_free(ui_t *ui);

// Updates the frame time and feeds a frame of mouse input to nuklear. Call once per frame,
// before any widget.
void ui_begin_frame(ui_t *ui);

// Builds the overlay windows: render statistics and settings, and the scene panel. Writes
// straight into `renderer->settings` and into the scene, both of which restart the
// accumulation by themselves. `selection` and `gizmo->mode` are read and written, because
// selection is made in the scene list and the mode has radio buttons there.
void ui_draw_overlay(ui_t *ui, renderer_t *renderer, gizmo_selection_t *selection,
                     gizmo_t *gizmo);

// Converts the frame's widgets to geometry and draws them over the swapchain image. Must
// come after whatever else writes that image, and always ends the nuklear frame.
void ui_record(ui_t *ui, gpu_frame_t *frame);

// True when a panel is under the cursor or a widget is being dragged. Everything else that
// wants the mouse -- the camera, the gizmo handles -- sits out while this holds. Only
// meaningful between ui_draw_overlay and the end of the frame, once the widgets have run.
bool ui_captures_mouse(const ui_t *ui);

// True while a text field has focus. Separate from the mouse, because typing continues
// wherever the cursor has wandered to, and WASD must not fly the camera while it does.
bool ui_captures_keyboard(const ui_t *ui);
