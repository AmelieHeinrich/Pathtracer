// The one place the nuklear feature set is chosen. Every translation unit that touches
// nuklear must agree on these defines -- the header configures its own ABI from them -- so
// nobody includes <nuklear.h> directly, they include this.
#pragma once

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_STANDARD_VARARGS // nk_labelf
// Turns the draw list into vertex and index buffers, which is the whole point of a GPU
// backend: without it nuklear only emits its high level command stream.
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
// Font baking pulls in nuklear's *own* copies of stb_truetype and stb_rect_pack, which is
// why the project's `stb` package stays uninvolved here.
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
// 32 bit nk_draw_index. The default 16 bit index silently corrupts the draw list once a
// frame exceeds 65535 vertices, and one uint32 index buffer costs nothing at this scale.
#define NK_UINT_DRAW_INDEX

#include <nuklear.h>
