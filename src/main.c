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
#include <stdlib.h>
#include <string.h>
#include <strings.h> // strcasecmp, so --tonemap accepts the names the overlay shows

static bool g_reload_requested = false;
// Like the reload and the pick below: set from the callback, serviced at the top of the loop,
// because renderer_screenshot stalls the device and must not run inside a recording frame.
static bool g_screenshot_requested = false;
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
    case GLFW_KEY_F12:
        g_screenshot_requested = true;
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

typedef struct pt_options_t {
    const char *scene;   // NULL: the default scene file
    const char *capture; // non-NULL: render `samples` samples into it, then exit
    uint32_t samples;
    // Negative leaves renderer_init's default alone. Overridable because sky_intensity is a
    // render setting rather than scene data, and cornell.pts is only itself with the sky at
    // 0 -- which also makes it the one scene a change to the sky cannot perturb.
    float sky;
    // Renders the material with no light transport at all. Worth a flag because it is the
    // renderer's own check on the spectral upsampling: an albedo turned into a spectrum and
    // resolved back must give the colour that was authored, and this is the view that shows
    // exactly that and nothing else.
    bool unlit;
    // The two variance reducers, both on by default, and both switchable off here for the
    // same reason --tonemap can be pinned: a capture is only comparable against another one
    // taken the same way. --no-adaptive matters most, because adaptive sampling turns
    // --samples from an exact per-pixel count into a per-pixel budget.
    bool no_clamp;
    bool no_adaptive;
    // Switches the a-trous filter on. The overlay is the usual way to reach it, but a
    // --capture runs headless to a file and never sees the overlay, so comparing a filtered
    // render against a raw one needs a flag.
    bool denoise;
    // Tonemapping is on by default, so unlike --denoise these exist to turn it *off* or pin
    // it: a --capture PNG is only comparable against another one shaped the same way. The
    // PFM path is unaffected either way -- it stops before the tonemapper by design.
    const char *tonemap; // NULL leaves the default curve
    float exposure;      // stops
    // Both negative to mean "leave the default alone", for the same reason --sky exists: the
    // sky is a render setting, so a capture that depends on it has to be able to pin it.
    float turbidity;
    float sun_elevation;
    // 0 leaves the default. Worth pinning because a transmissive object spends two bounces
    // per traversal, so a glass test is only comparable against a budget it cannot exhaust.
    uint32_t bounces;
    // Load, write straight back out, and exit. The only way to check from outside the UI that
    // every authored field survives a save -- which matters because the writer is what stands
    // between someone tuning a material in the panel and still having it tomorrow.
    const char *resave;
} pt_options_t;

// A deliberately tiny command line. Its whole reason for existing is --capture: comparing two
// builds means rendering the same scene from the same viewpoint to the *same sample count*
// and diffing the files, which is not something a human with a screenshot key can do
// repeatably.
//
// An unrecognised argument is an error rather than a warning, because the failure it guards
// against -- a comparison script quietly capturing under the wrong settings, and the diff
// being read as a real change -- is much worse than an early exit.
static bool parse_options(int argc, char **argv, pt_options_t *out)
{
    *out = (pt_options_t){
        .samples = 512u, .sky = -1.0f, .turbidity = -1.0f, .sun_elevation = -1000.0f};

    for (int i = 1; i < argc; ++i) {
        const bool has_value = i + 1 < argc;
        if (strcmp(argv[i], "--scene") == 0 && has_value) {
            out->scene = argv[++i];
        } else if (strcmp(argv[i], "--capture") == 0 && has_value) {
            out->capture = argv[++i];
        } else if (strcmp(argv[i], "--samples") == 0 && has_value) {
            out->samples = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--sky") == 0 && has_value) {
            out->sky = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--turbidity") == 0 && has_value) {
            out->turbidity = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--sun-elevation") == 0 && has_value) {
            out->sun_elevation = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--bounces") == 0 && has_value) {
            out->bounces = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--resave") == 0 && has_value) {
            out->resave = argv[++i];
        } else if (strcmp(argv[i], "--unlit") == 0) {
            out->unlit = true;
        } else if (strcmp(argv[i], "--no-clamp") == 0) {
            out->no_clamp = true;
        } else if (strcmp(argv[i], "--no-adaptive") == 0) {
            out->no_adaptive = true;
        } else if (strcmp(argv[i], "--denoise") == 0) {
            out->denoise = true;
        } else if (strcmp(argv[i], "--tonemap") == 0 && has_value) {
            out->tonemap = argv[++i];
        } else if (strcmp(argv[i], "--exposure") == 0 && has_value) {
            out->exposure = strtof(argv[++i], NULL);
        } else {
            fprintf(stderr,
                    "usage: %s [--scene FILE] [--sky N] [--unlit] [--denoise]"
                    " [--tonemap none|agx|aces|reinhard] [--exposure STOPS]"
                    " [--turbidity N] [--sun-elevation DEG] [--bounces N]"
                    " [--no-clamp] [--no-adaptive]"
                    " [--capture OUT.png|OUT.pfm [--samples N]]\n",
                    argv[0]);
            return false;
        }
    }

    // Zero would never satisfy the loop's exit test, leaving the window up forever with no
    // hint as to why.
    if (out->samples == 0u) {
        out->samples = 1u;
    }
    return true;
}

// Picks the first unused screenshot_NNNN.png in the working directory, which xmake's
// set_rundir pins to the repository root. It scans rather than counting from zero so a
// restart cannot overwrite what an earlier session captured.
static bool next_screenshot_path(char *out, size_t size)
{
    for (uint32_t i = 0; i < 10000u; ++i) {
        snprintf(out, size, "screenshot_%04u.png", i);
        FILE *existing = fopen(out, "rb");
        if (!existing) {
            return true;
        }
        fclose(existing);
    }

    fprintf(stderr, "screenshot: every name from screenshot_0000.png upwards is taken\n");
    return false;
}

// Installed before ui_init, which chains onto it: the wheel drives either the UI or the
// camera's move speed, and the loop below picks which by asking ui_captures_mouse.
static void on_scroll(GLFWwindow *window, double x, double y)
{
    (void)window;
    (void)x;
    g_scroll += (float)y;
}

int main(int argc, char **argv)
{
    pt_options_t options;
    if (!parse_options(argc, argv, &options)) {
        return 1;
    }

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
    pt_scene_load(&renderer.scene, options.scene ? options.scene : PT_SCENE_DIR "/models.pts");

    if (options.sky >= 0.0f) {
        renderer.settings.sky_intensity = options.sky;
    }
    if (options.unlit) {
        renderer.settings.flags |= PT_FLAG_UNLIT;
    }
    if (options.no_clamp) {
        renderer.settings.flags &= ~PT_FLAG_CLAMP;
    }
    if (options.no_adaptive) {
        renderer.settings.flags &= ~PT_FLAG_ADAPTIVE;
    }
    // Not members of renderer.settings, deliberately: both run over the averaged image and
    // must never restart the accumulation.
    renderer.denoise.settings.enabled = options.denoise;
    renderer.tonemap.settings.exposure = options.exposure;
    if (options.tonemap) {
        bool matched = false;
        for (uint32_t i = 0; i < (uint32_t)TONEMAP_CURVE_COUNT; ++i) {
            if (strcasecmp(options.tonemap, TONEMAP_CURVE_NAMES[i]) == 0) {
                renderer.tonemap.settings.curve = (tonemap_curve_t)i;
                matched = true;
                break;
            }
        }
        // An unrecognised curve is an error for the same reason an unrecognised argument is:
        // a capture script quietly shaping its output the wrong way is worse than an exit.
        if (!matched) {
            fprintf(stderr, "unknown tonemap curve '%s'\n", options.tonemap);
            return 1;
        }
    }
    if (options.resave) {
        const bool saved = pt_scene_save(&renderer.scene, options.resave);
        // Everything below needs a window and a device that are already up, so tearing down
        // properly would mean threading an early exit through all of it for a debug path.
        return saved ? 0 : 1;
    }
    if (options.bounces > 0u) {
        renderer.settings.max_bounces = options.bounces;
    }
    if (options.turbidity >= 0.0f) {
        renderer.settings.turbidity = options.turbidity;
    }
    if (options.sun_elevation > -999.0f) {
        renderer.settings.sun_elevation = options.sun_elevation;
    }

    pt_fly_camera_t fly;
    pt_fly_camera_init(&fly, renderer.scene.camera_position, renderer.scene.camera_yaw,
                       renderer.scene.camera_pitch);

    gizmo_t gizmo;
    gizmo_init(&gizmo);
    gizmo_selection_t selection = {0};

    printf("WASD/QE to fly, right mouse to look, wheel for speed.\n"
           "1/2/3 switch the gizmo between move, rotate and scale. R reloads the shaders.\n"
           "F12 writes the render to screenshot_NNNN.png, overlay excluded.\n");
    fflush(stdout);

    double last_time = glfwGetTime();
    bool mouse_was_down = false;
    // Capture mode's exit status. False until the image is written, so closing the window
    // early is reported as the failure it is rather than as a successful capture.
    bool capture_ok = false;
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

        if (g_screenshot_requested) {
            g_screenshot_requested = false;
            char path[64];
            if (next_screenshot_path(path, sizeof(path))) {
                renderer_screenshot(&renderer, path);
            }
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
            pt_fly_camera_update(&fly, window, dt, renderer.scene.camera_fov,
                                 renderer.scene.camera_aperture,
                                 renderer.scene.camera_focus_distance, scroll,
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

        // Capture mode: stop once the accumulator holds the requested number of samples.
        // Tested after the frame rather than before it, so the count reflects work that has
        // actually landed, and outside the frame because the readback stalls the device.
        if (options.capture) {
            const uint32_t taken =
                renderer.accum_frames * renderer.settings.samples_per_frame;
            if (taken >= options.samples) {
                // The extension picks the format: a .pfm for the diff script, anything else
                // for the eye.
                const char *dot = strrchr(options.capture, '.');
                capture_ok = dot && strcmp(dot, ".pfm") == 0
                                 ? renderer_capture_pfm(&renderer, options.capture)
                                 : renderer_screenshot(&renderer, options.capture);
                break;
            }
        }
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
    return options.capture && !capture_ok ? 1 : 0;
}
