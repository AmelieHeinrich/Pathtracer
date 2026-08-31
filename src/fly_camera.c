#include "fly_camera.h"

#include <GLFW/glfw3.h>

#include <math.h>
#include <string.h>

#define PT_DEG_TO_RAD (3.14159265358979323846f / 180.0f)
#define PT_LOOK_SENSITIVITY 0.12f // degrees per pixel
#define PT_PITCH_LIMIT 89.0f
#define PT_MIN_SPEED 0.05f
#define PT_MAX_SPEED 200.0f

void pt_fly_camera_init(pt_fly_camera_t *camera, const float position[3], float yaw,
                        float pitch)
{
    memset(camera, 0, sizeof(*camera));
    memcpy(camera->position, position, sizeof(camera->position));
    camera->yaw = yaw;
    camera->pitch = pitch;
    camera->move_speed = 4.0f;
}

static void forward_from_angles(const pt_fly_camera_t *camera, float out[3])
{
    const float yaw = camera->yaw * PT_DEG_TO_RAD;
    const float pitch = camera->pitch * PT_DEG_TO_RAD;
    const float cos_pitch = cosf(pitch);

    out[0] = cos_pitch * sinf(yaw);
    out[1] = sinf(pitch);
    out[2] = cos_pitch * cosf(yaw);
}

static bool key_held(GLFWwindow *window, int key)
{
    return glfwGetKey(window, key) == GLFW_PRESS;
}

pt_camera_t pt_fly_camera_update(pt_fly_camera_t *camera, GLFWwindow *window, float dt,
                                 float fov_degrees, float scroll, bool blocked)
{
    double cursor_x = 0.0;
    double cursor_y = 0.0;
    glfwGetCursorPos(window, &cursor_x, &cursor_y);

    const bool wants_look =
        !blocked && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

    if (wants_look && !camera->looking) {
        // Latch the anchor on the press rather than carrying a stale one, so re-entering
        // look mode after moving the cursor elsewhere does not snap the view.
        camera->last_cursor_x = cursor_x;
        camera->last_cursor_y = cursor_y;
    }
    camera->looking = wants_look;

    if (camera->looking) {
        const float dx = (float)(cursor_x - camera->last_cursor_x);
        const float dy = (float)(cursor_y - camera->last_cursor_y);
        camera->last_cursor_x = cursor_x;
        camera->last_cursor_y = cursor_y;

        // Both subtract: increasing yaw turns towards screen *left* under this basis, and
        // screen y grows downwards.
        camera->yaw -= dx * PT_LOOK_SENSITIVITY;
        camera->pitch -= dy * PT_LOOK_SENSITIVITY;

        camera->pitch = camera->pitch > PT_PITCH_LIMIT
                            ? PT_PITCH_LIMIT
                            : (camera->pitch < -PT_PITCH_LIMIT ? -PT_PITCH_LIMIT
                                                               : camera->pitch);
        // Kept in range so the value saved to a scene file stays readable.
        camera->yaw = fmodf(camera->yaw, 360.0f);
    }

    if (!blocked && scroll != 0.0f) {
        // Geometric, so one notch is the same proportional change at every speed.
        camera->move_speed *= powf(1.15f, scroll);
        camera->move_speed = camera->move_speed < PT_MIN_SPEED
                                 ? PT_MIN_SPEED
                                 : (camera->move_speed > PT_MAX_SPEED ? PT_MAX_SPEED
                                                                      : camera->move_speed);
    }

    float forward[3];
    forward_from_angles(camera, forward);

    // The same basis construction pt_camera_look_at performs, so W and D move along exactly
    // the axes the rendered image shows.
    const float world_up[3] = {0.0f, 1.0f, 0.0f};
    float right[3];
    pt_vec3_cross(right, forward, world_up);
    pt_vec3_normalize(right);

    if (!blocked) {
        float motion[3] = {0.0f, 0.0f, 0.0f};
        if (key_held(window, GLFW_KEY_W)) {
            pt_vec3_mad(motion, motion, forward, 1.0f);
        }
        if (key_held(window, GLFW_KEY_S)) {
            pt_vec3_mad(motion, motion, forward, -1.0f);
        }
        if (key_held(window, GLFW_KEY_D)) {
            pt_vec3_mad(motion, motion, right, 1.0f);
        }
        if (key_held(window, GLFW_KEY_A)) {
            pt_vec3_mad(motion, motion, right, -1.0f);
        }
        if (key_held(window, GLFW_KEY_E)) {
            pt_vec3_mad(motion, motion, world_up, 1.0f);
        }
        if (key_held(window, GLFW_KEY_Q)) {
            pt_vec3_mad(motion, motion, world_up, -1.0f);
        }

        // Normalised so moving diagonally is not faster than moving straight.
        if (pt_vec3_length(motion) > 0.0f) {
            pt_vec3_normalize(motion);
            float speed = camera->move_speed;
            if (key_held(window, GLFW_KEY_LEFT_SHIFT)) {
                speed *= 4.0f;
            }
            pt_vec3_mad(camera->position, camera->position, motion, speed * dt);
        }
    }

    float target[3];
    pt_vec3_add(target, camera->position, forward);
    return pt_camera_look_at(camera->position, target, fov_degrees);
}
