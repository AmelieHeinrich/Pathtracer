#pragma once

#include "gpu.h"
#include "pt_math.h"

// An immediate mode line list: everything queued between debug_draw_begin and
// debug_draw_record is drawn once and then forgotten, so callers describe what they want to
// see each frame rather than owning any geometry.
//
// Structurally a sibling of the nuklear backend in ui.c -- one pipeline, a mapped vertex
// buffer per frame in flight, and the same LOAD attachment over the presented image.

// Per list, and there are two of them. A ground grid and a set of gizmos is a few thousand
// vertices, so this is an order of magnitude of headroom; debug_draw_line drops rather than
// overruns.
#define DEBUG_DRAW_MAX_VERTICES 16384

// Colours are 0xRRGGBBAA, matching how they read in source.
#define DEBUG_COLOR_X 0xE04C4CFFu
#define DEBUG_COLOR_Y 0x4CE066FFu
#define DEBUG_COLOR_Z 0x4C8CE0FFu
#define DEBUG_COLOR_SELECTION 0xFFC24CFFu
#define DEBUG_COLOR_LIGHT 0xFFE88CFFu
#define DEBUG_COLOR_GRID 0x50505AFFu

typedef struct debug_vertex_t {
    float position[3]; // world space
    uint8_t color[4];
} debug_vertex_t;

typedef struct debug_draw_t {
    gpu_device_t *device; // not owned

    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkPipeline pipeline;

    gpu_buffer_t vertices[GPU_FRAMES_IN_FLIGHT];

    // Two lists, built up on the CPU during the frame and concatenated into the frame's
    // buffer at record time. They differ only in whether the scene may hide them, which is
    // one draw call each rather than any pipeline change.
    debug_vertex_t occluded[DEBUG_DRAW_MAX_VERTICES];
    uint32_t occluded_count;
    debug_vertex_t overlay[DEBUG_DRAW_MAX_VERTICES];
    uint32_t overlay_count;

    bool overlay_mode; // which list debug_draw_line is currently filling
    bool overflowed;   // latched, so a full list complains once rather than every frame
} debug_draw_t;

void debug_draw_init(debug_draw_t *debug, gpu_device_t *device, VkFormat color_format);
void debug_draw_free(debug_draw_t *debug);

// Drops everything queued last frame, and returns to occluded mode. Call once per frame
// before queueing anything.
void debug_draw_begin(debug_draw_t *debug);

// Chooses which list the calls after it go into. Overlay lines are drawn on top of the scene;
// occluded ones are hidden where geometry is in front of them. Anything that exists to be
// grabbed -- gizmo handles, light markers, the selection wireframe -- wants overlay; anything
// that describes the world, like the ground grid, wants to be occluded by it.
void debug_draw_set_overlay(debug_draw_t *debug, bool overlay);

void debug_draw_line(debug_draw_t *debug, const float a[3], const float b[3], uint32_t rgba);

// The 12 edges of a unit cube under `transform`, a row major 3x4 as pt_transform_compose
// returns. Used for entity bounds, so it follows rotation and non-uniform scale exactly.
void debug_draw_box(debug_draw_t *debug, const VkTransformMatrixKHR *transform, uint32_t rgba);

// Three axis-aligned rings, enough to read as a sphere without a real wireframe.
void debug_draw_sphere(debug_draw_t *debug, const float center[3], float radius,
                       uint32_t rgba);

// A ground plane grid on Y = 0, `extent` out from the origin in each direction.
void debug_draw_grid(debug_draw_t *debug, float extent, float step, uint32_t rgba);

// Draws the queued lines over the frame's image, occluded by `depth` -- the camera space
// depth image the path tracer wrote. Must come after whatever else drew into that image.
void debug_draw_record(debug_draw_t *debug, gpu_frame_t *frame, const pt_camera_t *camera,
                       const gpu_image_t *depth);
