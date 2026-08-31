#pragma once

#include "scene.h"

// A line based text format, chosen so a scene diffs cleanly in git and can be fixed up in a
// text editor without the program running. Blocks are introduced by a keyword at the start of
// a line; everything else is `key value...`; `#` begins a comment.
//
//   # pathtracer scene v1
//   camera
//     position 4.6 3.1 8.6
//     yaw -152
//     pitch -13
//     fov 40
//
//   entity "ground"
//     shape       plane
//     translation 0 0 0
//     rotation    0 0 0
//     scale       14 1 14
//     albedo      0.62 0.62 0.64
//     roughness   1
//
//   light "key"
//     type      point
//     position  0.5 4.4 2.6
//     color     1 0.92 0.78
//     intensity 60
//
// Only the authored half of pt_scene_t is touched; the GPU objects are left alone and a
// pt_scene_sync afterwards picks the change up through the bumped revision.

// Replaces the scene's authored data. Returns false and leaves the scene untouched when the
// file cannot be read; an unrecognised key inside a valid file only warns, so a scene written
// by a newer build still loads.
bool pt_scene_load(pt_scene_t *scene, const char *path);

bool pt_scene_save(const pt_scene_t *scene, const char *path);
