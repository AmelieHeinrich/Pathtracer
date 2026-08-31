#pragma once

#include "debug_draw.h"
#include "scene.h"

typedef struct GLFWwindow GLFWwindow;

typedef enum gizmo_target_t {
    GIZMO_TARGET_NONE = 0,
    GIZMO_TARGET_ENTITY,
    GIZMO_TARGET_LIGHT,
} gizmo_target_t;

// What the property panel is editing and what the manipulator is attached to. Selection is
// made in the UI list; there is no viewport picking.
typedef struct gizmo_selection_t {
    gizmo_target_t target;
    uint32_t index;
} gizmo_selection_t;

typedef enum gizmo_mode_t {
    GIZMO_MODE_TRANSLATE = 0,
    GIZMO_MODE_ROTATE,
    GIZMO_MODE_SCALE,
    GIZMO_MODE_COUNT,
} gizmo_mode_t;

typedef struct gizmo_t {
    gizmo_mode_t mode;

    // -1 when nothing is hovered or dragged; otherwise the axis, 0=X 1=Y 2=Z.
    int32_t hovered_axis;
    int32_t active_axis;

    // Captured on the press, so every frame of a drag is measured against where it started
    // rather than against the previous frame. That keeps a drag from accumulating drift and
    // makes it exactly reversible by returning the cursor.
    float drag_origin[3];      // the object's position when the drag began
    float drag_start_value[3]; // translation, rotation or scale at that moment
    float drag_start_offset;   // the axis parameter under the cursor then
    float drag_start_angle;    // screen space angle under the cursor then, for rotate
} gizmo_t;

void gizmo_init(gizmo_t *gizmo);

// Queues the scene annotations: a marker for every light and a wireframe on the selection,
// all drawn over the scene, plus an optional ground grid that sits in it.
void gizmo_draw_scene(debug_draw_t *debug, const pt_scene_t *scene, gizmo_selection_t selection,
                      bool show_grid, bool show_gizmos);

// Runs the manipulator for the current selection: hit tests the handles against the cursor,
// applies a drag, and queues the handle geometry. Bumps `scene->revision` on any change.
//
// `blocked` suppresses interaction, for when the UI owns the mouse. `visible` false hides the
// handles and stops them responding at all -- a handle nobody can see is not one anybody
// should be able to grab. Returns true while a drag is in progress, which the caller uses to
// keep the camera still.
bool gizmo_update(gizmo_t *gizmo, debug_draw_t *debug, pt_scene_t *scene,
                  gizmo_selection_t selection, const pt_camera_t *camera, GLFWwindow *window,
                  VkExtent2D viewport, bool blocked, bool visible);
