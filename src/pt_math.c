#include "pt_math.h"

#include <math.h>
#include <string.h>

#define PT_PI 3.14159265358979323846f
#define PT_DEG_TO_RAD (PT_PI / 180.0f)

// The depth range the debug projection maps onto [0,1]. Nothing samples this depth -- the
// gizmo occlusion test compares camera distances, not clip depth -- so the only requirement
// is that the whole scene stays inside it.
#define PT_NEAR_PLANE 0.05f
#define PT_FAR_PLANE 10000.0f

// ---------------------------------------------------------------------------
// vec3
// ---------------------------------------------------------------------------

void pt_vec3_add(float out[3], const float a[3], const float b[3])
{
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
}

void pt_vec3_sub(float out[3], const float a[3], const float b[3])
{
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

void pt_vec3_scale(float out[3], const float v[3], float s)
{
    out[0] = v[0] * s;
    out[1] = v[1] * s;
    out[2] = v[2] * s;
}

void pt_vec3_mad(float out[3], const float a[3], const float b[3], float s)
{
    out[0] = a[0] + b[0] * s;
    out[1] = a[1] + b[1] * s;
    out[2] = a[2] + b[2] * s;
}

float pt_vec3_dot(const float a[3], const float b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void pt_vec3_cross(float out[3], const float a[3], const float b[3])
{
    // Written through temporaries so `out` may alias `a` or `b`.
    const float x = a[1] * b[2] - a[2] * b[1];
    const float y = a[2] * b[0] - a[0] * b[2];
    const float z = a[0] * b[1] - a[1] * b[0];
    out[0] = x;
    out[1] = y;
    out[2] = z;
}

float pt_vec3_length(const float v[3])
{
    return sqrtf(pt_vec3_dot(v, v));
}

float pt_vec3_distance(const float a[3], const float b[3])
{
    float delta[3];
    pt_vec3_sub(delta, a, b);
    return pt_vec3_length(delta);
}

void pt_vec3_normalize(float v[3])
{
    const float length = pt_vec3_length(v);
    if (length > 0.0f) {
        pt_vec3_scale(v, v, 1.0f / length);
    }
}

// ---------------------------------------------------------------------------
// mat4
// ---------------------------------------------------------------------------

void pt_mat4_transform(float out[4], const pt_mat4_t *m, const float v[3])
{
    for (uint32_t row = 0; row < 4; ++row) {
        out[row] = m->m[row][0] * v[0] + m->m[row][1] * v[1] + m->m[row][2] * v[2] +
                   m->m[row][3];
    }
}

// ---------------------------------------------------------------------------
// camera
// ---------------------------------------------------------------------------

pt_camera_t pt_camera_look_at(const float position[3], const float target[3],
                              float fov_degrees)
{
    pt_camera_t camera = {0};
    memcpy(camera.position, position, sizeof(camera.position));

    pt_vec3_sub(camera.forward, target, position);
    pt_vec3_normalize(camera.forward);

    const float world_up[3] = {0.0f, 1.0f, 0.0f};
    pt_vec3_cross(camera.right, camera.forward, world_up);
    pt_vec3_normalize(camera.right);
    pt_vec3_cross(camera.up, camera.right, camera.forward);

    camera.tan_half_fov = tanf(fov_degrees * 0.5f * PT_DEG_TO_RAD);
    return camera;
}

pt_mat4_t pt_camera_view_projection(const pt_camera_t *camera, float aspect)
{
    // Derived to agree with raygen exactly. There, a pixel's ray direction is
    //     forward + (ndc.x * aspect * tan) * right - (ndc.y * tan) * up
    // so for a point at camera space (vx, vy, vz) = (dot(v,right), dot(v,up), dot(v,forward)),
    //     ndc.x = vx / (vz * tan * aspect)      ndc.y = -vy / (vz * tan)
    // which is the usual perspective divide by vz with a Y flip, Vulkan's NDC having y = -1
    // at the top of the framebuffer just as raygen's does.
    const float f = 1.0f / camera->tan_half_fov;
    const float x_scale = f / aspect;

    // Standard [0,1] depth mapping: z' = a*vz + b, w' = vz.
    const float a = PT_FAR_PLANE / (PT_FAR_PLANE - PT_NEAR_PLANE);
    const float b = -PT_FAR_PLANE * PT_NEAR_PLANE / (PT_FAR_PLANE - PT_NEAR_PLANE);

    const float *right = camera->right;
    const float *up = camera->up;
    const float *forward = camera->forward;
    const float *eye = camera->position;

    // Each row folds the world-to-camera projection of one axis together with the
    // translation, hence the -dot(axis, eye) in the last column.
    pt_mat4_t out;
    out.m[0][0] = x_scale * right[0];
    out.m[0][1] = x_scale * right[1];
    out.m[0][2] = x_scale * right[2];
    out.m[0][3] = -x_scale * pt_vec3_dot(right, eye);

    out.m[1][0] = -f * up[0];
    out.m[1][1] = -f * up[1];
    out.m[1][2] = -f * up[2];
    out.m[1][3] = f * pt_vec3_dot(up, eye);

    out.m[2][0] = a * forward[0];
    out.m[2][1] = a * forward[1];
    out.m[2][2] = a * forward[2];
    out.m[2][3] = -a * pt_vec3_dot(forward, eye) + b;

    out.m[3][0] = forward[0];
    out.m[3][1] = forward[1];
    out.m[3][2] = forward[2];
    out.m[3][3] = -pt_vec3_dot(forward, eye);

    return out;
}

bool pt_project_to_screen(const pt_mat4_t *view_projection, const float world[3],
                          float viewport_width, float viewport_height, float out_screen[2])
{
    float clip[4];
    pt_mat4_transform(clip, view_projection, world);

    // w is the camera space depth, so this is exactly "behind or on the eye plane".
    if (clip[3] <= 1e-5f) {
        return false;
    }

    // NDC runs -1..1 across the framebuffer in both axes, y already flipped by the matrix.
    out_screen[0] = (clip[0] / clip[3] * 0.5f + 0.5f) * viewport_width;
    out_screen[1] = (clip[1] / clip[3] * 0.5f + 0.5f) * viewport_height;
    return true;
}

void pt_screen_to_ray(const pt_camera_t *camera, float aspect, float screen_x, float screen_y,
                      float viewport_width, float viewport_height, float out_origin[3],
                      float out_direction[3])
{
    // The inverse of the projection above, written the way raygen writes it rather than as a
    // matrix inverse, so the two cannot drift apart.
    const float ndc_x = (screen_x / viewport_width) * 2.0f - 1.0f;
    const float ndc_y = (screen_y / viewport_height) * 2.0f - 1.0f;

    memcpy(out_origin, camera->position, sizeof(float) * 3);

    memcpy(out_direction, camera->forward, sizeof(float) * 3);
    pt_vec3_mad(out_direction, out_direction, camera->right,
                ndc_x * aspect * camera->tan_half_fov);
    pt_vec3_mad(out_direction, out_direction, camera->up, -ndc_y * camera->tan_half_fov);
    pt_vec3_normalize(out_direction);
}

// ---------------------------------------------------------------------------
// geometry
// ---------------------------------------------------------------------------

bool pt_ray_plane(const float origin[3], const float direction[3], const float plane_point[3],
                  const float plane_normal[3], float *out_t)
{
    const float denominator = pt_vec3_dot(direction, plane_normal);
    // Near-parallel: the intersection is either nowhere or unusably far away.
    if (fabsf(denominator) < 1e-6f) {
        return false;
    }

    float to_plane[3];
    pt_vec3_sub(to_plane, plane_point, origin);
    const float t = pt_vec3_dot(to_plane, plane_normal) / denominator;
    if (t <= 0.0f) {
        return false;
    }

    *out_t = t;
    return true;
}

float pt_point_segment_distance(const float p[2], const float a[2], const float b[2])
{
    const float abx = b[0] - a[0];
    const float aby = b[1] - a[1];
    const float length_squared = abx * abx + aby * aby;

    // Degenerate segment: fall back to the distance to its single point.
    float t = 0.0f;
    if (length_squared > 1e-12f) {
        t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / length_squared;
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    }

    const float dx = p[0] - (a[0] + abx * t);
    const float dy = p[1] - (a[1] + aby * t);
    return sqrtf(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// transforms
// ---------------------------------------------------------------------------

// Rows of the XYZ Euler rotation, R = Rz * Ry * Rx, so X is applied to the object first.
static void rotation_rows(const float rotation_degrees[3], float out[3][3])
{
    const float cx = cosf(rotation_degrees[0] * PT_DEG_TO_RAD);
    const float sx = sinf(rotation_degrees[0] * PT_DEG_TO_RAD);
    const float cy = cosf(rotation_degrees[1] * PT_DEG_TO_RAD);
    const float sy = sinf(rotation_degrees[1] * PT_DEG_TO_RAD);
    const float cz = cosf(rotation_degrees[2] * PT_DEG_TO_RAD);
    const float sz = sinf(rotation_degrees[2] * PT_DEG_TO_RAD);

    out[0][0] = cz * cy;
    out[0][1] = cz * sy * sx - sz * cx;
    out[0][2] = cz * sy * cx + sz * sx;

    out[1][0] = sz * cy;
    out[1][1] = sz * sy * sx + cz * cx;
    out[1][2] = sz * sy * cx - cz * sx;

    out[2][0] = -sy;
    out[2][1] = cy * sx;
    out[2][2] = cy * cx;
}

VkTransformMatrixKHR pt_transform_compose(const float translation[3], const float rotation[3],
                                          const float scale[3])
{
    float rows[3][3];
    rotation_rows(rotation, rows);

    // R * S scales along the object's own axes, which is what a per axis scale in a property
    // panel is expected to mean; the translation then rides in the last column.
    VkTransformMatrixKHR out;
    for (uint32_t row = 0; row < 3; ++row) {
        for (uint32_t col = 0; col < 3; ++col) {
            out.matrix[row][col] = rows[row][col] * scale[col];
        }
        out.matrix[row][3] = translation[row];
    }
    return out;
}

void pt_transform_axis(const float rotation[3], uint32_t axis, float out[3])
{
    float rows[3][3];
    rotation_rows(rotation, rows);

    // Column `axis` of the rotation is where that object axis ends up in world space.
    for (uint32_t row = 0; row < 3; ++row) {
        out[row] = rows[row][axis];
    }
}
