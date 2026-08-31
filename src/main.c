#include "gpu.h" // must precede glfw3.h so it exposes its Vulkan entry points
#include "debug_draw.h"
#include "fly_camera.h"
#include "gizmo.h"
#include "renderer.h"
#include "scene_io.h"
#include "ui.h"

#include <GLFW/glfw3.h>

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static bool g_reload_requested = false;
// Accumulated by the callback and drained once a frame, so a fast wheel never loses notches.
static float g_scroll = 0.0f;
// Set by the number keys; the UI has radio buttons for the same thing. 1/2/3 rather than the
// usual W/E/R because those are all taken here -- W and E move the camera and R reloads
// the shaders.
static gizmo_mode_t g_requested_mode = GIZMO_MODE_TRANSLATE;

static void on_key(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    (void)window;
    (void)scancode;
    (void)mods;

    if (action != GLFW_PRESS) {
        return;
    }

    switch (key) {
    case GLFW_KEY_R:
        g_reload_requested = true;
        break;
    case GLFW_KEY_1:
        g_requested_mode = GIZMO_MODE_TRANSLATE;
        break;
    case GLFW_KEY_2:
        g_requested_mode = GIZMO_MODE_ROTATE;
        break;
    case GLFW_KEY_3:
        g_requested_mode = GIZMO_MODE_SCALE;
        break;
    default:
        break;
    }
}

// Installed before ui_init, which chains onto it: the wheel drives either the UI or the
// camera's move speed, and the loop below picks which by asking ui_captures_mouse.
static void on_scroll(GLFWwindow *window, double x, double y)
{
    (void)window;
    (void)x;
    g_scroll += (float)y;
}

int main(void)
{
    // volk owns the loader, so GLFW has to be handed vkGetInstanceProcAddr rather than
    // dlopen'ing libvulkan itself. Both must happen before glfwInit().
    if (volkInitialize() != VK_SUCCESS) {
        fprintf(stderr, "no Vulkan loader found\n");
        return 1;
    }
    glfwInitVulkanLoader(vkGetInstanceProcAddr);

    if (!glfwInit()) {
        fprintf(stderr, "failed to initialize GLFW\n");
        return 1;
    }
    if (!glfwVulkanSupported()) {
        fprintf(stderr, "GLFW reports no Vulkan support on this platform\n");
        glfwTerminate();
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // no OpenGL context for a Vulkan window
    GLFWwindow *window = glfwCreateWindow(1280, 720, "Pathtracer", NULL, NULL);
    if (!window) {
        fprintf(stderr, "failed to create window\n");
        glfwTerminate();
        return 1;
    }
    glfwSetKeyCallback(window, on_key);
    glfwSetScrollCallback(window, on_scroll);

    gpu_device_t device;
    gpu_swapchain_t swapchain;
    gpu_frames_t frames;
    renderer_t renderer;
    debug_draw_t debug;
    ui_t ui;

    gpu_device_init(&device);
    gpu_swapchain_init(&swapchain, &device, window);
    gpu_frames_init(&frames, &device);
    renderer_init(&renderer, &device);
    // Both overlays draw into the swapchain image, so their pipelines need that format. It is
    // picked deterministically from the surface, so a recreate never changes it.
    debug_draw_init(&debug, &device, swapchain.format);
    ui_init(&ui, &device, window, swapchain.format);

    // Falls back to the built-in showcase renderer_init already put in place.
    pt_scene_load(&renderer.scene, PT_SCENE_DIR "/default.pts");

    pt_fly_camera_t fly;
    pt_fly_camera_init(&fly, renderer.scene.camera_position, renderer.scene.camera_yaw,
                       renderer.scene.camera_pitch);

    gizmo_t gizmo;
    gizmo_init(&gizmo);
    gizmo_selection_t selection = {0};

    printf("WASD/QE to fly, right mouse to look, wheel for speed.\n"
           "1/2/3 switch the gizmo between move, rotate and scale. R reloads the shaders.\n");
    fflush(stdout);

    double last_time = glfwGetTime();
    bool mouse_was_down = false;
    // Set when a click lands in the viewport; serviced at the top of the next iteration,
    // because renderer_pick stalls the device and must not run inside a recording frame.
    bool pick_pending = false;
    uint32_t pick_x = 0;
    uint32_t pick_y = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (pick_pending) {
            pick_pending = false;
            const uint32_t hit = renderer_pick(&renderer, pick_x, pick_y);
            if (hit == UINT32_MAX) {
                selection.target = GIZMO_TARGET_NONE;
            } else {
                selection.target = GIZMO_TARGET_ENTITY;
                selection.index = hit;
            }
        }

        const double now = glfwGetTime();
        const float dt = (float)(now - last_time);
        last_time = now;

        if (g_reload_requested) {
            g_reload_requested = false;
            renderer_reload_shaders(&renderer);
        }

        // Rebuilds the acceleration structure when the scene was edited. Outside the frame,
        // because it submits and waits on its own.
        renderer_sync_scene(&renderer);

        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if (width == 0 || height == 0) {
            glfwWaitEvents(); // minimised: block rather than spin
            continue;
        }

        gpu_frame_t *frame = gpu_frame_begin(&frames, &swapchain);
        if (!frame) {
            continue;
        }

        renderer_resize(&renderer, frame->extent);

        // Built only once the frame is known to be good, so a skipped frame can never leave
        // half a nuklear frame behind. The overlay runs before renderer_record so a click
        // this frame takes effect in this frame's dispatch.
        ui_begin_frame(&ui);
        gizmo.mode = g_requested_mode;
        ui_draw_overlay(&ui, &renderer, &selection, &gizmo);
        g_requested_mode = gizmo.mode; // the panel's radio buttons can change it too

        // Only meaningful now that the widgets have run.
        const bool ui_has_mouse = ui_captures_mouse(&ui);

        // The camera looks with the right button and the gizmo drags with the left, so the
        // two cannot fight over a press; blocking on the drag as well only keeps WASD from
        // sliding the view out from under a handle mid-drag.
        const float scroll = g_scroll;
        g_scroll = 0.0f;
        const pt_camera_t camera =
            pt_fly_camera_update(&fly, window, dt, renderer.scene.camera_fov, scroll,
                                 ui_captures_keyboard(&ui) || gizmo.active_axis >= 0);

        // Kept in step so saving the scene captures wherever the camera ended up. Not an
        // edit, so it deliberately does not bump the revision.
        memcpy(renderer.scene.camera_position, fly.position,
               sizeof(renderer.scene.camera_position));
        renderer.scene.camera_yaw = fly.yaw;
        renderer.scene.camera_pitch = fly.pitch;

        debug_draw_begin(&debug);
        gizmo_draw_scene(&debug, &renderer.scene, selection, ui.show_grid, ui.show_gizmos);
        const bool gizmo_busy = gizmo_update(&gizmo, &debug, &renderer.scene, selection,
                                             &camera, window, frame->extent, ui_has_mouse,
                                             ui.show_gizmos);

        // A click in the viewport selects whatever is under it -- but only one the gizmo did
        // not take for a handle, and only on the press, so holding the button does not
        // re-pick every frame.
        const bool mouse_down =
            glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        if (mouse_down && !mouse_was_down && !ui_has_mouse && !gizmo_busy) {
            double cursor_x = 0.0;
            double cursor_y = 0.0;
            glfwGetCursorPos(window, &cursor_x, &cursor_y);

            int window_width = 0;
            int window_height = 0;
            glfwGetWindowSize(window, &window_width, &window_height);
            if (window_width > 0 && window_height > 0) {
                // The cursor is in logical units; the pick image is in framebuffer pixels.
                pick_x = (uint32_t)(cursor_x * (double)frame->extent.width / window_width);
                pick_y = (uint32_t)(cursor_y * (double)frame->extent.height / window_height);
                pick_pending = true;
            }
        }
        mouse_was_down = mouse_down;

        renderer_record(&renderer, frame, &camera);
        // Lines first, then nuklear, so a panel is never drawn under a gizmo.
        debug_draw_record(&debug, frame, &camera, &renderer.depth);
        ui_record(&ui, frame);

        gpu_frame_end(&frames, &swapchain, frame);
    }

    // Nothing below may destroy an object the GPU could still be using.
    vkDeviceWaitIdle(device.device);

    ui_free(&ui);
    debug_draw_free(&debug);
    renderer_free(&renderer);
    gpu_frames_free(&frames);
    gpu_swapchain_free(&swapchain);
    gpu_device_free(&device);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
