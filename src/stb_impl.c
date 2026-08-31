// stb_image_write's single-header implementation, in a translation unit of its own with
// warnings off -- the same arrangement nuklear_impl.c gets, and for the same reason: it does
// not compile clean under -Wall and that is no reason to lower the bar for src/.
//
// Only PNG is wanted in either direction, so every other codec is compiled out. The renderer
// uses the writer for one thing -- writing the output image to a file, see
// renderer_screenshot -- and the reader for one thing, loading the blue noise tiles in
// src/bluenoise.c.

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WRITE_NO_STDIO_DEPRECATION
#define STBI_ONLY_PNG

#include <stb/stb_image_write.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
