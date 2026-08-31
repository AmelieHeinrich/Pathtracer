#include "gizmo.h"

#include <GLFW/glfw3.h>

#include <math.h>
#include <string.h>

#define GIZMO_PI 3.14159265358979323846f
#define GIZMO_DEG_TO_RAD (GIZMO_PI / 180.0f)
#define GIZMO_RAD_TO_DEG (180.0f / GIZMO_PI)

// The handle is kept this long on screen whatever the distance, so it stays grabbable when
// the camera is far away and does not swallow the object when it is close.
#define GIZMO_HANDLE_PIXELS 90.0f
#define GIZMO_PICK_PIXELS 9.0f
#define GIZMO_RING_SEGMENTS 32

#define GIZMO_COLOR_HOVER 0xFFFFFFFFu

void gizmo_init(gizmo_t *gizmo)
{
    memset(gizmo, 0, sizeof(*gizmo));
    gizmo->hovered_axis = -1;
    gizmo->active_axis = -1;
}

// ---------------------------------------------------------------------------
// scene annotations
// ---------------------------------------------------------------------------

// Any unit vector perpendicular to `v`. Picking the axis `v` is least aligned with keeps the
// cross product well conditioned, the same trick the integrator uses to build a tangent.
static void perpendicular(const float v[3], float out[3])
{
    const float helper[3] = {fabsf(v[0]) > 0.9f ? 0.0f : 1.0f, fabsf(v[0]) > 0.9f ? 1.0f : 0.0f,
                             0.0f};
    pt_vec3_cross(out, v, helper);
    pt_vec3_normalize(out);
}

static void draw_light_marker(debug_draw_t *debug, const pt_light_t *light, uint32_t color)
{
    debug_draw_sphere(debug, light->position, 0.18f, color);

    float direction[3];
    memcpy(direction, light->direction, sizeof(direction));
    if (pt_vec3_length(direction) < 1e-6f) {
        direction[0] = 0.0f;
        direction[1] = -1.0f;
        direction[2] = 0.0f;
    }
    pt_vec3_normalize(direction);

    if (light->type == PT_LIGHT_POINT) {
        // A six-pointed star: enough to read as "radiates in every direction".
        for (uint32_t axis = 0; axis < 3; ++axis) {
            for (int32_t sign = -1; sign <= 1; sign += 2) {
                float tip[3];
                memcpy(tip, light->position, sizeof(tip));
                tip[axis] += 0.45f * (float)sign;
                debug_draw_line(debug, light->position, tip, color);
            }
        }
        return;
    }

    if (light->type == PT_LIGHT_DIRECTIONAL) {
        float tip[3];
        pt_vec3_mad(tip, light->position, direction, 1.6f);
        debug_draw_line(debug, light->position, tip, color);

        // Four barbs folded back from the tip, so the arrow reads from any angle.
        float u[3];
        float v[3];
        perpendicular(direction, u);
        pt_vec3_cross(v, direction, u);
        for (uint32_t i = 0; i < 4; ++i) {
            const float *side = (i % 2 == 0) ? u : v;
            const float sign = (i < 2) ? 1.0f : -1.0f;

            float barb[3];
            pt_vec3_mad(barb, tip, direction, -0.3f);
            pt_vec3_mad(barb, barb, side, 0.15f * sign);
            debug_draw_line(debug, tip, barb, color);
        }
        return;
    }

    if (light->type == PT_LIGHT_AREA) {
        // The emitting rectangle itself, plus a stub normal so which way it faces is not a
        // guess. The tangents are derived the same way fill_light derives them, so the
        // outline lands exactly on the surface that is being sampled.
        float u[3];
        float v[3];
        perpendicular(direction, u);
        pt_vec3_cross(v, direction, u);

        float corners[4][3];
        const float signs[4][2] = {{-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 1.0f}};
        for (uint32_t i = 0; i < 4; ++i) {
            memcpy(corners[i], light->position, sizeof(corners[i]));
            pt_vec3_mad(corners[i], corners[i], u, signs[i][0] * light->size[0]);
            pt_vec3_mad(corners[i], corners[i], v, signs[i][1] * light->size[1]);
        }
        for (uint32_t i = 0; i < 4; ++i) {
            debug_draw_line(debug, corners[i], corners[(i + 1) % 4], color);
        }

        float tip[3];
        pt_vec3_mad(tip, light->position, direction, 0.6f);
        debug_draw_line(debug, light->position, tip, color);
        return;
    }

    // Spot: four edges of the outer cone plus the disc they end on, which is the only way to
    // see the cone angle without rendering it.
    const float length = 2.5f;
    const float radius = length * tanf(light->cone_outer * GIZMO_DEG_TO_RAD);

    float u[3];
    float v[3];
    perpendicular(direction, u);
    pt_vec3_cross(v, direction, u);

    float center[3];
    pt_vec3_mad(center, light->position, direction, length);

    float previous[3];
    for (uint32_t i = 0; i <= GIZMO_RING_SEGMENTS; ++i) {
        const float angle = (float)i / GIZMO_RING_SEGMENTS * 2.0f * GIZMO_PI;

        float point[3];
        pt_vec3_mad(point, center, u, cosf(angle) * radius);
        pt_vec3_mad(point, point, v, sinf(angle) * radius);

        if (i > 0) {
            debug_draw_line(debug, previous, point, color);
        }
        if (i % (GIZMO_RING_SEGMENTS / 4) == 0 && i < GIZMO_RING_SEGMENTS) {
            debug_draw_line(debug, light->position, point, color);
        }
        memcpy(previous, point, sizeof(previous));
    }
}

void gizmo_draw_scene(debug_draw_t *debug, const pt_scene_t *scene, gizmo_selection_t selection,
                      bool show_grid, bool show_gizmos)
{
    // The grid describes the ground, so it belongs in the scene and is hidden by it.
    debug_draw_set_overlay(debug, false);
    if (show_grid) {
        debug_draw_grid(debug, 20.0f, 1.0f, DEBUG_COLOR_GRID);
    }

    if (!show_gizmos) {
        return;
    }

    // Everything below is a gizmo: it exists to be seen and grabbed, so nothing hides it.
    debug_draw_set_overlay(debug, true);

    for (uint32_t i = 0; i < scene->light_count; ++i) {
        const bool selected =
            selection.target == GIZMO_TARGET_LIGHT && selection.index == i;
        draw_light_marker(debug, &scene->lights[i],
                          selected ? DEBUG_COLOR_SELECTION : DEBUG_COLOR_LIGHT);
    }

    // Only the selection gets a wireframe. Boxing every entity turns a scene of any size into
    // a thicket, and the ground plane's box alone would span the whole view.
    if (selection.target == GIZMO_TARGET_ENTITY && selection.index < scene->entity_count) {
        const pt_entity_t *entity = &scene->entities[selection.index];
        const VkTransformMatrixKHR transform =
            pt_transform_compose(entity->translation, entity->rotation, entity->scale);
        debug_draw_box(debug, &transform, DEBUG_COLOR_SELECTION);
    }
}

// ---------------------------------------------------------------------------
// manipulator
// ---------------------------------------------------------------------------

// The three float triples a manipulator can edit. Lights have no rotation or scale, so those
// come back NULL and the caller falls back to translating.
typedef struct bindings_t {
    float *translation;
    float *rotation;
    float *scale;
} bindings_t;

static bindings_t selection_bindings(pt_scene_t *scene, gizmo_selection_t selection)
{
    bindings_t out = {0};

    if (selection.target == GIZMO_TARGET_ENTITY && selection.index < scene->entity_count) {
        pt_entity_t *entity = &scene->entities[selection.index];
        out.translation = entity->translation;
        out.rotation = entity->rotation;
        out.scale = entity->scale;
    } else if (selection.target == GIZMO_TARGET_LIGHT && selection.index < scene->light_count) {
        // A directional light ignores its position when shading, but it still anchors the
        // marker, so moving it is meaningful as a piece of scene furniture.
        out.translation = scene->lights[selection.index].position;
    }

    return out;
}

// The world axes the handles point along: the object's own, so a scale handle scales the
// axis it is drawn on and a translate handle follows a rotated object.
static void axis_directions(const bindings_t *bindings, float axes[3][3])
{
    for (uint32_t i = 0; i < 3; ++i) {
        if (bindings->rotation) {
            pt_transform_axis(bindings->rotation, i, axes[i]);
        } else {
            axes[i][0] = i == 0 ? 1.0f : 0.0f;
            axes[i][1] = i == 1 ? 1.0f : 0.0f;
            axes[i][2] = i == 2 ? 1.0f : 0.0f;
        }
    }
}

static uint32_t axis_color(uint32_t axis)
{
    const uint32_t colors[3] = {DEBUG_COLOR_X, DEBUG_COLOR_Y, DEBUG_COLOR_Z};
    return colors[axis];
}

// Looking within this angle of an axis leaves too little of the view direction perpendicular
// to it to build a stable drag plane from. sin(8.6 degrees).
#define GIZMO_MIN_PLANE_SIN 0.15f
// Well past the far side of any plausible scene. A drag that computes an offset larger than
// this has degenerated, and letting it through would put a non-finite transform in the
// acceleration structure and fault the GPU.
#define GIZMO_MAX_OFFSET 1.0e5f

// The plane a translate or scale drag is measured on: it contains the axis, and its normal is
// whatever component of the view direction is perpendicular to that axis. That keeps the
// plane as square to the camera as it can be while still containing the axis.
//
// Returns false when the axis points too nearly at the camera for that to leave anything to
// normalise. Dragging along an axis aimed at the eye is meaningless anyway -- the cursor
// barely moves for any amount of travel -- so refusing is both safe and correct.
static bool drag_plane_normal(const float axis[3], const float view_direction[3], float out[3])
{
    const float along = pt_vec3_dot(view_direction, axis);
    pt_vec3_mad(out, view_direction, axis, -along);

    // Both inputs are unit, so this length is the sine of the angle between them.
    if (pt_vec3_length(out) < GIZMO_MIN_PLANE_SIN) {
        return false;
    }

    pt_vec3_scale(out, out, -1.0f);
    pt_vec3_normalize(out);
    return true;
}

// Where along `axis` the cursor currently sits, measured from `origin`. False when the cursor
// ray cannot reach the drag plane at all.
static bool axis_offset_under_cursor(const pt_camera_t *camera, float aspect,
                                     const float cursor[2], VkExtent2D viewport,
                                     const float origin[3], const float axis[3],
                                     float *out_offset)
{
    float ray_origin[3];
    float ray_direction[3];
    pt_screen_to_ray(camera, aspect, cursor[0], cursor[1], (float)viewport.width,
                     (float)viewport.height, ray_origin, ray_direction);

    float view_direction[3];
    pt_vec3_sub(view_direction, origin, camera->position);
    pt_vec3_normalize(view_direction);

    float normal[3];
    if (!drag_plane_normal(axis, view_direction, normal)) {
        return false;
    }

    float t;
    if (!pt_ray_plane(ray_origin, ray_direction, origin, normal, &t)) {
        return false;
    }

    float hit[3];
    pt_vec3_mad(hit, ray_origin, ray_direction, t);
    pt_vec3_sub(hit, hit, origin);
    const float offset = pt_vec3_dot(hit, axis);

    // The last line of defence before this number reaches an instance transform.
    if (!isfinite(offset) || fabsf(offset) > GIZMO_MAX_OFFSET) {
        return false;
    }

    *out_offset = offset;
    return true;
}

// Wraps into (-180, 180]. A single rotate drag is therefore limited to half a turn, which is
// the price of measuring against where the drag started rather than against the last frame --
// and measuring against the start is what makes the drag exactly reversible.
static float wrap_degrees(float degrees)
{
    while (degrees > 180.0f) {
        degrees -= 360.0f;
    }
    while (degrees <= -180.0f) {
        degrees += 360.0f;
    }
    return degrees;
}

// Queues one axis handle and reports how close the cursor came to it in pixels.
static float draw_and_measure_handle(debug_draw_t *debug, const pt_mat4_t *view_projection,
                                     VkExtent2D viewport, const float origin[3],
                                     const float axis[3], float length, gizmo_mode_t mode,
                                     const float cursor[2], uint32_t color)
{
    const float viewport_width = (float)viewport.width;
    const float viewport_height = (float)viewport.height;
    float best = 1e30f;

    if (mode == GIZMO_MODE_ROTATE) {
        // A ring in the plane the axis is normal to, hit tested against the same segments it
        // is drawn from.
        float u[3];
        float v[3];
        perpendicular(axis, u);
        pt_vec3_cross(v, axis, u);

        float previous_world[3];
        float previous_screen[2];
        bool previous_valid = false;

        for (uint32_t i = 0; i <= GIZMO_RING_SEGMENTS; ++i) {
            const float angle = (float)i / GIZMO_RING_SEGMENTS * 2.0f * GIZMO_PI;

            float point[3];
            pt_vec3_mad(point, origin, u, cosf(angle) * length);
            pt_vec3_mad(point, point, v, sinf(angle) * length);

            float screen[2];
            const bool valid = pt_project_to_screen(view_projection, point, viewport_width,
                                                    viewport_height, screen);
            if (i > 0 && valid && previous_valid) {
                debug_draw_line(debug, previous_world, point, color);
                const float distance = pt_point_segment_distance(cursor, previous_screen,
                                                                 screen);
                best = distance < best ? distance : best;
            }
            memcpy(previous_world, point, sizeof(previous_world));
            memcpy(previous_screen, screen, sizeof(previous_screen));
            previous_valid = valid;
        }
        return best;
    }

    float tip[3];
    pt_vec3_mad(tip, origin, axis, length);
    debug_draw_line(debug, origin, tip, color);

    if (mode == GIZMO_MODE_SCALE) {
        // A box at the tip, the usual shorthand for "this one scales".
        const VkTransformMatrixKHR head = pt_transform_compose(
            tip, (float[3]){0.0f, 0.0f, 0.0f},
            (float[3]){length * 0.06f, length * 0.06f, length * 0.06f});
        debug_draw_box(debug, &head, color);
    } else {
        // An open arrowhead, drawn in the plane most square to the axis.
        float u[3];
        float v[3];
        perpendicular(axis, u);
        pt_vec3_cross(v, axis, u);
        for (uint32_t i = 0; i < 4; ++i) {
            const float *side = (i % 2 == 0) ? u : v;
            const float sign = (i < 2) ? 1.0f : -1.0f;

            float barb[3];
            pt_vec3_mad(barb, tip, axis, -length * 0.18f);
            pt_vec3_mad(barb, barb, side, length * 0.08f * sign);
            debug_draw_line(debug, tip, barb, color);
        }
    }

    float origin_screen[2];
    float tip_screen[2];
    if (pt_project_to_screen(view_projection, origin, viewport_width, viewport_height,
                             origin_screen) &&
        pt_project_to_screen(view_projection, tip, viewport_width, viewport_height,
                             tip_screen)) {
        best = pt_point_segment_distance(cursor, origin_screen, tip_screen);
    }
    return best;
}

bool gizmo_update(gizmo_t *gizmo, debug_draw_t *debug, pt_scene_t *scene,
                  gizmo_selection_t selection, const pt_camera_t *camera, GLFWwindow *window,
                  VkExtent2D viewport, bool blocked, bool visible)
{
    const bindings_t bindings = selection_bindings(scene, selection);
    // Hiding the handles clears any drag in progress through this same path, so a gizmo
    // switched off mid-drag cannot leave the camera blocked on an axis nobody can release.
    if (!visible || !bindings.translation || viewport.width == 0 || viewport.height == 0) {
        gizmo->hovered_axis = -1;
        gizmo->active_axis = -1;
        return false;
    }

    // Lights carry only a position, so the other two modes have nothing to act on.
    gizmo_mode_t mode = gizmo->mode;
    if ((mode == GIZMO_MODE_ROTATE && !bindings.rotation) ||
        (mode == GIZMO_MODE_SCALE && !bindings.scale)) {
        mode = GIZMO_MODE_TRANSLATE;
    }

    // The cursor arrives in logical window units; everything here is in framebuffer pixels.
    int window_width = 0;
    int window_height = 0;
    glfwGetWindowSize(window, &window_width, &window_height);
    if (window_width <= 0 || window_height <= 0) {
        return false;
    }

    double cursor_x = 0.0;
    double cursor_y = 0.0;
    glfwGetCursorPos(window, &cursor_x, &cursor_y);
    const float cursor[2] = {
        (float)cursor_x * (float)viewport.width / (float)window_width,
        (float)cursor_y * (float)viewport.height / (float)window_height,
    };

    const float aspect = (float)viewport.width / (float)viewport.height;
    const pt_mat4_t view_projection = pt_camera_view_projection(camera, aspect);

    // Handles are always on top. A handle you cannot see is a handle you cannot grab, and the
    // geometry hiding it is usually the very thing being manipulated.
    debug_draw_set_overlay(debug, true);

    const float *origin = bindings.translation;

    // Constant apparent size: one pixel is this many world units at the object's depth, and
    // the handle is a fixed number of pixels long.
    float to_object[3];
    pt_vec3_sub(to_object, origin, camera->position);
    const float depth = pt_vec3_dot(to_object, camera->forward);
    if (depth < 0.01f) {
        gizmo->hovered_axis = -1;
        gizmo->active_axis = -1;
        return false; // behind the camera
    }
    const float world_per_pixel =
        2.0f * depth * camera->tan_half_fov / (float)viewport.height;
    const float handle_length = GIZMO_HANDLE_PIXELS * world_per_pixel;

    float axes[3][3];
    axis_directions(&bindings, axes);

    const bool mouse_down =
        glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;

    // A drag that has already started keeps running even over the UI; only *starting* one is
    // blocked, so dragging a handle under a panel does not abruptly stop.
    if (gizmo->active_axis >= 0 && !mouse_down) {
        gizmo->active_axis = -1;
    }

    // Hit test and draw in one pass, so the geometry measured is exactly the geometry shown.
    int32_t closest = -1;
    float closest_distance = GIZMO_PICK_PIXELS;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        const bool highlighted =
            gizmo->active_axis == (int32_t)axis ||
            (gizmo->active_axis < 0 && gizmo->hovered_axis == (int32_t)axis);
        const float distance = draw_and_measure_handle(
            debug, &view_projection, viewport, origin, axes[axis], handle_length, mode, cursor,
            highlighted ? GIZMO_COLOR_HOVER : axis_color(axis));

        if (distance < closest_distance) {
            closest_distance = distance;
            closest = (int32_t)axis;
        }
    }
    // While a drag is running the highlight stays on the dragged axis, not on whatever the
    // cursor happens to be near.
    if (gizmo->active_axis < 0) {
        gizmo->hovered_axis = blocked ? -1 : closest;
    }

    // --- begin a drag ---
    if (gizmo->active_axis < 0 && mouse_down && !blocked && closest >= 0) {
        const float *axis = axes[closest];

        float offset = 0.0f;
        bool ok = true;
        if (mode == GIZMO_MODE_ROTATE) {
            float origin_screen[2];
            ok = pt_project_to_screen(&view_projection, origin, (float)viewport.width,
                                      (float)viewport.height, origin_screen);
            if (ok) {
                gizmo->drag_start_angle = atan2f(cursor[1] - origin_screen[1],
                                                 cursor[0] - origin_screen[0]) *
                                          GIZMO_RAD_TO_DEG;
            }
        } else {
            ok = axis_offset_under_cursor(camera, aspect, cursor, viewport, origin, axis,
                                          &offset);
        }

        if (ok) {
            gizmo->active_axis = closest;
            gizmo->drag_start_offset = offset;
            memcpy(gizmo->drag_origin, origin, sizeof(gizmo->drag_origin));

            const float *source = mode == GIZMO_MODE_TRANSLATE ? bindings.translation
                                  : mode == GIZMO_MODE_ROTATE  ? bindings.rotation
                                                               : bindings.scale;
            memcpy(gizmo->drag_start_value, source, sizeof(gizmo->drag_start_value));
        }
    }

    if (gizmo->active_axis < 0) {
        return false;
    }

    // --- apply the drag ---
    // Everything is measured against the state captured on the press, so the drag never
    // accumulates error and returning the cursor returns the object.
    const uint32_t axis_index = (uint32_t)gizmo->active_axis;
    const float *axis = axes[axis_index];

    if (mode == GIZMO_MODE_ROTATE) {
        float origin_screen[2];
        if (pt_project_to_screen(&view_projection, gizmo->drag_origin, (float)viewport.width,
                                 (float)viewport.height, origin_screen)) {
            const float angle = atan2f(cursor[1] - origin_screen[1],
                                       cursor[0] - origin_screen[0]) *
                                GIZMO_RAD_TO_DEG;
            // Flip when the axis points away from the camera, so a clockwise drag always
            // turns the object clockwise as seen on screen.
            float to_camera[3];
            pt_vec3_sub(to_camera, camera->position, gizmo->drag_origin);
            const float sign = pt_vec3_dot(axis, to_camera) > 0.0f ? -1.0f : 1.0f;

            const float delta = wrap_degrees(angle - gizmo->drag_start_angle) * sign;
            // Added straight to the Euler angle rather than composed as a true rotation about
            // the axis. With Euler storage the two only agree when the other two angles are
            // zero, and adding is the one that round-trips exactly through the numbers the
            // property panel shows.
            bindings.rotation[axis_index] = gizmo->drag_start_value[axis_index] + delta;
            ++scene->revision;
        }
        return true;
    }

    float offset = 0.0f;
    if (!axis_offset_under_cursor(camera, aspect, cursor, viewport, gizmo->drag_origin, axis,
                                  &offset)) {
        return true; // still dragging, just nothing usable under the cursor this frame
    }
    const float delta = offset - gizmo->drag_start_offset;

    if (mode == GIZMO_MODE_TRANSLATE) {
        pt_vec3_mad(bindings.translation, gizmo->drag_start_value, axis, delta);
    } else {
        // Floored rather than clamped to zero: a zero scale collapses the instance transform
        // and the acceleration structure build has nothing left to work with.
        const float scaled = gizmo->drag_start_value[axis_index] + delta;
        bindings.scale[axis_index] = scaled < 0.01f ? 0.01f : scaled;
    }
    ++scene->revision;

    return true;
}
