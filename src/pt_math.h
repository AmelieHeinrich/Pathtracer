// Vector, matrix and camera maths shared by the renderer, the debug lines and the gizmo.
#pragma once

#include "gpu.h" // VkTransformMatrixKHR

#include <stdbool.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// vec3
// ---------------------------------------------------------------------------

void pt_vec3_add(float out[3], const float a[3], const float b[3]);
void pt_vec3_sub(float out[3], const float a[3], const float b[3]);
void pt_vec3_scale(float out[3], const float v[3], float s);
// out = a + b * s. The one compound form that shows up everywhere below.
void pt_vec3_mad(float out[3], const float a[3], const float b[3], float s);
float pt_vec3_dot(const float a[3], const float b[3]);
void pt_vec3_cross(float out[3], const float a[3], const float b[3]);
float pt_vec3_length(const float v[3]);
float pt_vec3_distance(const float a[3], const float b[3]);
// No-op on a zero vector rather than producing NaNs.
void pt_vec3_normalize(float v[3]);

// ---------------------------------------------------------------------------
// mat4
// ---------------------------------------------------------------------------

// Row major: m[row][col], so a transform is `out[i] = dot(m[i], v)`. Shaders take this as
// four float4 rows and do those dot products by hand, which keeps the whole HLSL/SPIR-V
// matrix layout question from ever arising.
typedef struct pt_mat4_t {
    float m[4][4];
} pt_mat4_t;

// out = m * v, with v treated as a point (w = 1). Returns the full homogeneous result.
void pt_mat4_transform(float out[4], const pt_mat4_t *m, const float v[3]);

// ---------------------------------------------------------------------------
// camera
// ---------------------------------------------------------------------------

// Matches `struct Camera` in shaders/pathtracer.slang under scalar layout. The basis is
// built on the CPU so the renderer can tell when it moved and restart the accumulation.
typedef struct pt_camera_t {
    float position[3];
    float forward[3];
    float right[3];
    float up[3];
    float tan_half_fov;
} pt_camera_t;

// Points the camera at `target` from `position` with a vertical field of view in degrees.
pt_camera_t pt_camera_look_at(const float position[3], const float target[3],
                              float fov_degrees);

// Builds the same projection raygen traces with, as a matrix: Vulkan clip space, Y down,
// depth in [0,1]. This is what makes debug lines land exactly on the geometry they annotate.
pt_mat4_t pt_camera_view_projection(const pt_camera_t *camera, float aspect);

// Projects a world point to framebuffer pixels. Returns false when it is behind the camera,
// in which case `out_screen` is untouched.
bool pt_project_to_screen(const pt_mat4_t *view_projection, const float world[3],
                         float viewport_width, float viewport_height, float out_screen[2]);

// The world space ray through a framebuffer pixel, matching raygen's ray for that pixel.
void pt_screen_to_ray(const pt_camera_t *camera, float aspect, float screen_x, float screen_y,
                      float viewport_width, float viewport_height, float out_origin[3],
                      float out_direction[3]);

// ---------------------------------------------------------------------------
// geometry
// ---------------------------------------------------------------------------

// Returns false when the ray is parallel to the plane or would hit it behind the origin.
bool pt_ray_plane(const float origin[3], const float direction[3], const float plane_point[3],
                  const float plane_normal[3], float *out_t);

// Distance from `p` to the segment `a`-`b`, in whatever 2D space all three share.
float pt_point_segment_distance(const float p[2], const float a[2], const float b[2]);

// ---------------------------------------------------------------------------
// transforms
// ---------------------------------------------------------------------------

// Translation, XYZ Euler rotation in degrees (X applied first, then Y, then Z) and a per axis
// scale, composed into the row major 3x4 an acceleration structure instance takes.
VkTransformMatrixKHR pt_transform_compose(const float translation[3], const float rotation[3],
                                          const float scale[3]);

// The rotated unit axis `axis` (0=X, 1=Y, 2=Z) of that same rotation, for orienting gizmo
// handles and wireframes without rebuilding the whole matrix.
void pt_transform_axis(const float rotation[3], uint32_t axis, float out[3]);
