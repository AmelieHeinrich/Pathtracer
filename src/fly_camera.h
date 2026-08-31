#pragma once

#include "pt_math.h"

typedef struct GLFWwindow GLFWwindow;

// A first person camera driven straight from GLFW input, replacing the fixed viewpoint and
// orbit the program used to have. It owns yaw and pitch rather than a basis, because those
// are what a scene file can round-trip and what mouse look actually moves; the basis is
// rebuilt through pt_camera_look_at each frame so the tracer and the debug lines can never
// disagree about where the camera is.
typedef struct pt_fly_camera_t {
    float position[3];
    // Degrees. forward = (cos(pitch) sin(yaw), sin(pitch), cos(pitch) cos(yaw)), so yaw 0
    // looks down +Z. Pitch is clamped short of vertical to keep the basis well conditioned.
    float yaw;
    float pitch;
    float move_speed; // units per second

    bool looking; // right mouse held
    double last_cursor_x;
    double last_cursor_y;
} pt_fly_camera_t;

void pt_fly_camera_init(pt_fly_camera_t *camera, const float position[3], float yaw,
                        float pitch);

// Applies a frame of input and returns the camera to render with. `dt` is in seconds.
// `blocked` suppresses input entirely, for when the UI or a gizmo owns the mouse; the camera
// is still returned so the caller always has one. `scroll` adjusts the movement speed.
pt_camera_t pt_fly_camera_update(pt_fly_camera_t *camera, GLFWwindow *window, float dt,
                                 float fov_degrees, float scroll, bool blocked);
