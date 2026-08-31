#include "scene_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PT_LINE_MAX 512
#define PT_KEY_MAX 32
#define PT_SPACE " \t\r\n"

// ---------------------------------------------------------------------------
// reading
// ---------------------------------------------------------------------------

typedef enum block_t {
    BLOCK_NONE = 0,
    BLOCK_CAMERA,
    BLOCK_ENTITY,
    BLOCK_LIGHT,
} block_t;

// Pulls the name out of `entity "ground"`. Taking everything between the *first* and *last*
// quote means a name containing quotes survives a round trip, and an unnamed block is simply
// left with whatever default it was created with.
static void parse_quoted_name(const char *rest, char *out, size_t out_size)
{
    const char *open = strchr(rest, '"');
    const char *close = open ? strrchr(rest, '"') : NULL;
    if (!open || close <= open) {
        return;
    }

    size_t length = (size_t)(close - open - 1);
    if (length >= out_size) {
        length = out_size - 1;
    }
    // The whole buffer, not just up to the terminator: leaving the tail as whatever the
    // default name happened to put there would make two loads of the same file differ
    // bytewise, which is a trap for anything that ever compares these structs.
    memset(out, 0, out_size);
    memcpy(out, open + 1, length);
}

// Consumes up to `max` whitespace separated floats from the running strtok state. Values the
// line does not supply are left as they were, so a short line is a partial edit rather than
// a silent zeroing.
static void read_floats(float *out, uint32_t max)
{
    for (uint32_t i = 0; i < max; ++i) {
        const char *token = strtok(NULL, PT_SPACE);
        if (!token) {
            return;
        }
        out[i] = strtof(token, NULL);
    }
}

bool pt_scene_load(pt_scene_t *scene, const char *path)
{
    FILE *file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "scene: cannot open '%s'\n", path);
        return false;
    }

    // Past this point the scene is being replaced. Anything unparseable only warns, so a
    // file written by a newer build still loads as much as this one understands.
    scene->entity_count = 0;
    scene->light_count = 0;

    block_t block = BLOCK_NONE;
    pt_entity_t *entity = NULL;
    pt_light_t *light = NULL;
    uint32_t line_number = 0;

    char line[PT_LINE_MAX];
    while (fgets(line, sizeof(line), file)) {
        ++line_number;

        char *comment = strchr(line, '#');
        if (comment) {
            *comment = '\0';
        }

        char *cursor = line + strspn(line, PT_SPACE);
        if (*cursor == '\0') {
            continue;
        }

        // Split the leading keyword off by hand rather than with strtok, because the block
        // lines need the untokenised remainder to find their quoted name in.
        const size_t key_length = strcspn(cursor, PT_SPACE);
        char key[PT_KEY_MAX];
        if (key_length >= sizeof(key)) {
            fprintf(stderr, "scene: %s:%u: key too long\n", path, line_number);
            continue;
        }
        memcpy(key, cursor, key_length);
        key[key_length] = '\0';

        char *rest = cursor + key_length;

        if (strcmp(key, "camera") == 0) {
            block = BLOCK_CAMERA;
            continue;
        }
        if (strcmp(key, "entity") == 0) {
            // The shape is not known yet, so this starts as a cube and the `shape` line
            // below corrects it.
            entity = pt_scene_add_entity(scene, PT_SHAPE_CUBE);
            block = entity ? BLOCK_ENTITY : BLOCK_NONE;
            if (entity) {
                parse_quoted_name(rest, entity->name, sizeof(entity->name));
            }
            continue;
        }
        if (strcmp(key, "light") == 0) {
            light = pt_scene_add_light(scene, PT_LIGHT_POINT);
            block = light ? BLOCK_LIGHT : BLOCK_NONE;
            if (light) {
                parse_quoted_name(rest, light->name, sizeof(light->name));
            }
            continue;
        }

        // Everything else is a key/value inside the current block. This first token is the
        // one every key needs; the multi-value keys pick the rest up through read_floats,
        // which walks the same strtok state onwards.
        char *value = strtok(rest, PT_SPACE);
        if (!value) {
            fprintf(stderr, "scene: %s:%u: '%s' has no value\n", path, line_number, key);
            continue;
        }

        bool handled = false;

        if (block == BLOCK_CAMERA) {
            handled = true;
            if (strcmp(key, "position") == 0) {
                scene->camera_position[0] = strtof(value, NULL);
                read_floats(&scene->camera_position[1], 2);
            } else if (strcmp(key, "yaw") == 0) {
                scene->camera_yaw = strtof(value, NULL);
            } else if (strcmp(key, "pitch") == 0) {
                scene->camera_pitch = strtof(value, NULL);
            } else if (strcmp(key, "fov") == 0) {
                scene->camera_fov = strtof(value, NULL);
            } else if (strcmp(key, "aperture") == 0) {
                scene->camera_aperture = strtof(value, NULL);
            } else if (strcmp(key, "focus_distance") == 0) {
                scene->camera_focus_distance = strtof(value, NULL);
            } else {
                handled = false;
            }
        } else if (block == BLOCK_ENTITY && entity) {
            handled = true;
            if (strcmp(key, "shape") == 0) {
                pt_shape_t shape;
                if (pt_shape_from_name(value, &shape)) {
                    entity->shape = shape;
                } else {
                    fprintf(stderr, "scene: %s:%u: unknown shape '%s'\n", path, line_number,
                            value);
                }
            } else if (strcmp(key, "translation") == 0) {
                entity->translation[0] = strtof(value, NULL);
                read_floats(&entity->translation[1], 2);
            } else if (strcmp(key, "rotation") == 0) {
                entity->rotation[0] = strtof(value, NULL);
                read_floats(&entity->rotation[1], 2);
            } else if (strcmp(key, "scale") == 0) {
                entity->scale[0] = strtof(value, NULL);
                read_floats(&entity->scale[1], 2);
            } else if (strcmp(key, "albedo") == 0) {
                entity->albedo[0] = strtof(value, NULL);
                read_floats(&entity->albedo[1], 2);
            } else if (strcmp(key, "emission") == 0) {
                entity->emission[0] = strtof(value, NULL);
                read_floats(&entity->emission[1], 2);
            } else if (strcmp(key, "emission_strength") == 0) {
                entity->emission_strength = strtof(value, NULL);
            } else if (strcmp(key, "roughness") == 0) {
                entity->roughness = strtof(value, NULL);
            } else if (strcmp(key, "metallic") == 0) {
                entity->metallic = strtof(value, NULL);
            } else if (strcmp(key, "transmission") == 0) {
                entity->transmission = strtof(value, NULL);
            } else if (strcmp(key, "ior") == 0) {
                entity->ior = strtof(value, NULL);
            } else if (strcmp(key, "abbe") == 0) {
                entity->abbe = strtof(value, NULL);
            } else if (strcmp(key, "absorption") == 0) {
                entity->absorption[0] = strtof(value, NULL);
                read_floats(&entity->absorption[1], 2);
            } else if (strcmp(key, "absorption_distance") == 0) {
                entity->absorption_distance = strtof(value, NULL);
            } else {
                handled = false;
            }
        } else if (block == BLOCK_LIGHT && light) {
            handled = true;
            if (strcmp(key, "type") == 0) {
                pt_light_type_t type;
                if (pt_light_type_from_name(value, &type)) {
                    light->type = type;
                } else {
                    fprintf(stderr, "scene: %s:%u: unknown light type '%s'\n", path,
                            line_number, value);
                }
            } else if (strcmp(key, "position") == 0) {
                light->position[0] = strtof(value, NULL);
                read_floats(&light->position[1], 2);
            } else if (strcmp(key, "direction") == 0) {
                light->direction[0] = strtof(value, NULL);
                read_floats(&light->direction[1], 2);
            } else if (strcmp(key, "color") == 0) {
                light->color[0] = strtof(value, NULL);
                read_floats(&light->color[1], 2);
            } else if (strcmp(key, "intensity") == 0) {
                light->intensity = strtof(value, NULL);
            } else if (strcmp(key, "temperature") == 0) {
                light->temperature = strtof(value, NULL);
            } else if (strcmp(key, "range") == 0) {
                light->range = strtof(value, NULL);
            } else if (strcmp(key, "cone") == 0) {
                light->cone_inner = strtof(value, NULL);
                read_floats(&light->cone_outer, 1);
            } else if (strcmp(key, "size") == 0) {
                light->size[0] = strtof(value, NULL);
                read_floats(&light->size[1], 1);
            } else {
                handled = false;
            }
        }

        if (!handled) {
            fprintf(stderr, "scene: %s:%u: ignoring unknown key '%s'\n", path, line_number,
                    key);
        }
    }

    fclose(file);

    // Every add above already bumped it, but an empty file must still force a resync.
    ++scene->revision;
    printf("scene: loaded %s (%u entities, %u lights)\n", path, scene->entity_count,
           scene->light_count);
    fflush(stdout);
    return true;
}

// ---------------------------------------------------------------------------
// writing
// ---------------------------------------------------------------------------

bool pt_scene_save(const pt_scene_t *scene, const char *path)
{
    FILE *file = fopen(path, "w");
    if (!file) {
        fprintf(stderr, "scene: cannot write '%s'\n", path);
        return false;
    }

    // %g throughout: short enough to stay readable and diffable, and round trips every value
    // the UI can actually produce.
    fprintf(file, "# pathtracer scene v1\n\n");

    fprintf(file, "camera\n");
    fprintf(file, "  position %g %g %g\n", (double)scene->camera_position[0],
            (double)scene->camera_position[1], (double)scene->camera_position[2]);
    fprintf(file, "  yaw %g\n", (double)scene->camera_yaw);
    fprintf(file, "  pitch %g\n", (double)scene->camera_pitch);
    fprintf(file, "  fov %g\n", (double)scene->camera_fov);
    fprintf(file, "  aperture %g\n", (double)scene->camera_aperture);
    fprintf(file, "  focus_distance %g\n", (double)scene->camera_focus_distance);

    for (uint32_t i = 0; i < scene->entity_count; ++i) {
        const pt_entity_t *entity = &scene->entities[i];
        fprintf(file, "\nentity \"%s\"\n", entity->name);
        fprintf(file, "  shape       %s\n", pt_shape_name(entity->shape));
        fprintf(file, "  translation %g %g %g\n", (double)entity->translation[0],
                (double)entity->translation[1], (double)entity->translation[2]);
        fprintf(file, "  rotation    %g %g %g\n", (double)entity->rotation[0],
                (double)entity->rotation[1], (double)entity->rotation[2]);
        fprintf(file, "  scale       %g %g %g\n", (double)entity->scale[0],
                (double)entity->scale[1], (double)entity->scale[2]);
        fprintf(file, "  albedo      %g %g %g\n", (double)entity->albedo[0],
                (double)entity->albedo[1], (double)entity->albedo[2]);
        // Only written when it is doing something, so an ordinary surface stays two lines
        // shorter and the ones that glow stand out in a diff.
        if (entity->emission_strength != 0.0f) {
            fprintf(file, "  emission    %g %g %g\n", (double)entity->emission[0],
                    (double)entity->emission[1], (double)entity->emission[2]);
            fprintf(file, "  emission_strength %g\n", (double)entity->emission_strength);
        }
        fprintf(file, "  roughness   %g\n", (double)entity->roughness);
        // Each written only when it is doing something, exactly as emission is: an ordinary
        // opaque surface stays as short as it was before materials existed, and a scene file
        // diff shows only the entities that actually gained a property.
        if (entity->metallic != 0.0f) {
            fprintf(file, "  metallic    %g\n", (double)entity->metallic);
        }
        if (entity->transmission != 0.0f) {
            fprintf(file, "  transmission %g\n", (double)entity->transmission);
            fprintf(file, "  ior         %g\n", (double)entity->ior);
            if (entity->abbe != 0.0f) {
                fprintf(file, "  abbe        %g\n", (double)entity->abbe);
            }
            if (entity->absorption[0] != 1.0f || entity->absorption[1] != 1.0f ||
                entity->absorption[2] != 1.0f) {
                fprintf(file, "  absorption  %g %g %g\n", (double)entity->absorption[0],
                        (double)entity->absorption[1], (double)entity->absorption[2]);
                fprintf(file, "  absorption_distance %g\n",
                        (double)entity->absorption_distance);
            }
        }
    }

    for (uint32_t i = 0; i < scene->light_count; ++i) {
        const pt_light_t *light = &scene->lights[i];
        fprintf(file, "\nlight \"%s\"\n", light->name);
        fprintf(file, "  type      %s\n", pt_light_type_name(light->type));
        // Both written whatever the type, even though a point light ignores its direction and
        // a directional one ignores its position: the gizmo anchors on the position and the
        // panel keeps both, so omitting either would lose an edit across a save and reload.
        fprintf(file, "  position  %g %g %g\n", (double)light->position[0],
                (double)light->position[1], (double)light->position[2]);
        fprintf(file, "  direction %g %g %g\n", (double)light->direction[0],
                (double)light->direction[1], (double)light->direction[2]);
        fprintf(file, "  color     %g %g %g\n", (double)light->color[0],
                (double)light->color[1], (double)light->color[2]);
        fprintf(file, "  intensity %g\n", (double)light->intensity);
        // Only when set, matching how emission and range are written: an ordinary light
        // stays one line shorter and a warm one stands out in a diff.
        if (light->temperature != 0.0f) {
            fprintf(file, "  temperature %g\n", (double)light->temperature);
        }
        if (light->range != 0.0f) {
            fprintf(file, "  range     %g\n", (double)light->range);
        }
        if (light->type == PT_LIGHT_SPOT) {
            fprintf(file, "  cone      %g %g\n", (double)light->cone_inner,
                    (double)light->cone_outer);
        }
        if (light->type == PT_LIGHT_AREA) {
            fprintf(file, "  size      %g %g\n", (double)light->size[0],
                    (double)light->size[1]);
        }
    }

    const bool ok = ferror(file) == 0;
    fclose(file);

    if (ok) {
        printf("scene: saved %s\n", path);
        fflush(stdout);
    } else {
        fprintf(stderr, "scene: write failed for '%s'\n", path);
    }
    return ok;
}
