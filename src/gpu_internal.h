// Shared plumbing between the gpu_* translation units. Not part of the public API.
#pragma once

#include "gpu.h"

#include <stdio.h>
#include <stdlib.h>

const char *vk_result_string(VkResult result);

// Prints to stderr and exits. Never returns.
void gpu_fatal(const char *fmt, ...);

// malloc that aborts instead of returning NULL.
void *gpu_alloc(size_t size);

#define PT_COUNT(a) ((uint32_t)(sizeof(a) / sizeof((a)[0])))

// Unlike assert(), this always evaluates `expr` -- release builds define NDEBUG.
#define VK_CHECK(expr)                                                                   \
    do {                                                                                 \
        VkResult pt_res_ = (expr);                                                       \
        if (pt_res_ != VK_SUCCESS) {                                                     \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr,         \
                    vk_result_string(pt_res_));                                          \
            abort();                                                                     \
        }                                                                                \
    } while (0)
