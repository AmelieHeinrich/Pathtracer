#include "gpu_internal.h"

#include <GLFW/glfw3.h>

#include <stdarg.h>
#include <string.h>

// Everything below assumes Vulkan 1.4 core, so dynamic rendering, synchronization2,
// timeline semaphores, buffer device address, descriptor indexing, maintenance5 and
// push descriptors need no extension strings -- only feature bits.
#define PT_API_VERSION VK_API_VERSION_1_4

static const char *const PT_DEVICE_EXTENSIONS[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    // Lets images live in VK_IMAGE_LAYOUT_GENERAL with no performance penalty, so the
    // only transition left in the frame loop is the one for presentation.
    VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME,
};

static const char *const PT_VALIDATION_LAYER = "VK_LAYER_KHRONOS_validation";

// ---------------------------------------------------------------------------
// diagnostics
// ---------------------------------------------------------------------------

const char *vk_result_string(VkResult result)
{
    switch (result) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
    case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    case VK_ERROR_FRAGMENTATION: return "VK_ERROR_FRAGMENTATION";
    case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
    case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
    case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR: return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
    case VK_ERROR_VALIDATION_FAILED_EXT: return "VK_ERROR_VALIDATION_FAILED_EXT";
    default: return "VK_ERROR_<unknown>";
    }
}

void gpu_fatal(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    fputs("gpu: fatal: ", stderr);
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
    va_end(args);
    exit(EXIT_FAILURE);
}

static VKAPI_ATTR VkBool32 VKAPI_CALL gpu_debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT types,
    const VkDebugUtilsMessengerCallbackDataEXT *data,
    void *user_data)
{
    (void)types;
    (void)user_data;

    const char *label = "info";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        label = "error";
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        label = "warning";
    }

    fprintf(stderr, "vulkan %s: %s\n", label, data->pMessage ? data->pMessage : "(null)");
    return VK_FALSE;
}

static VkDebugUtilsMessengerCreateInfoEXT gpu_debug_messenger_info(void)
{
    return (VkDebugUtilsMessengerCreateInfoEXT){
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = gpu_debug_callback,
    };
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

static bool has_extension(const VkExtensionProperties *props, uint32_t count, const char *name)
{
    for (uint32_t i = 0; i < count; ++i) {
        if (strcmp(props[i].extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

static bool has_layer(const VkLayerProperties *props, uint32_t count, const char *name)
{
    for (uint32_t i = 0; i < count; ++i) {
        if (strcmp(props[i].layerName, name) == 0) {
            return true;
        }
    }
    return false;
}

void *gpu_alloc(size_t size)
{
    void *p = malloc(size ? size : 1);
    if (!p) {
        gpu_fatal("out of memory");
    }
    return p;
}

// ---------------------------------------------------------------------------
// feature set
//
// One list drives both the "does this device support it" check and the "turn it on"
// path at vkCreateDevice, so the two can never drift apart.
// ---------------------------------------------------------------------------

#define GPU_REQUIRED_FEATURES(X)                                                         \
    X(base.features, shaderInt64)                                                        \
    X(base.features, samplerAnisotropy)                                                  \
    X(base.features, shaderStorageImageWriteWithoutFormat)                               \
    X(base.features, shaderStorageImageReadWithoutFormat)                                \
    X(v12, descriptorIndexing)                                                           \
    X(v12, shaderSampledImageArrayNonUniformIndexing)                                    \
    X(v12, shaderStorageBufferArrayNonUniformIndexing)                                   \
    X(v12, descriptorBindingSampledImageUpdateAfterBind)                                 \
    X(v12, descriptorBindingPartiallyBound)                                              \
    X(v12, descriptorBindingVariableDescriptorCount)                                     \
    X(v12, runtimeDescriptorArray)                                                       \
    X(v12, scalarBlockLayout)                                                            \
    X(v12, hostQueryReset)                                                               \
    X(v12, timelineSemaphore)                                                            \
    X(v12, bufferDeviceAddress)                                                          \
    X(v13, synchronization2)                                                             \
    X(v13, dynamicRendering)                                                             \
    X(v13, maintenance4)                                                                 \
    X(v14, maintenance5)                                                                 \
    X(v14, pushDescriptor)                                                               \
    X(accel, accelerationStructure)                                                      \
    X(rt_pipeline, rayTracingPipeline)                                                   \
    X(ray_query, rayQuery)                                                               \
    X(unified_layouts, unifiedImageLayouts)

typedef struct gpu_features_t {
    VkPhysicalDeviceFeatures2 base;
    VkPhysicalDeviceVulkan11Features v11;
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan13Features v13;
    VkPhysicalDeviceVulkan14Features v14;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel;
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline;
    VkPhysicalDeviceRayQueryFeaturesKHR ray_query;
    VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR unified_layouts;
} gpu_features_t;

// Zeroes the struct and links the pNext chain. Valid both for querying (all bits
// cleared, filled in by the driver) and for creating (bits set by gpu_features_enable).
static void gpu_features_chain(gpu_features_t *g)
{
    memset(g, 0, sizeof(*g));

    g->base.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    g->v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    g->v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    g->v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    g->v14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    g->accel.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    g->rt_pipeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    g->ray_query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    g->unified_layouts.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR;

    g->base.pNext = &g->v11;
    g->v11.pNext = &g->v12;
    g->v12.pNext = &g->v13;
    g->v13.pNext = &g->v14;
    g->v14.pNext = &g->accel;
    g->accel.pNext = &g->rt_pipeline;
    g->rt_pipeline.pNext = &g->ray_query;
    g->ray_query.pNext = &g->unified_layouts;
    g->unified_layouts.pNext = NULL;
}

// Returns the number of unsupported features, writing up to `max` of their names to `out`.
static uint32_t gpu_features_missing(const gpu_features_t *g, const char **out, uint32_t max)
{
    uint32_t count = 0;
#define X(member, field)                                                                 \
    if (!g->member.field) {                                                              \
        if (count < max) {                                                               \
            out[count] = #field;                                                         \
        }                                                                                \
        ++count;                                                                         \
    }
    GPU_REQUIRED_FEATURES(X)
#undef X
    return count;
}

// Deliberately enables only what we require -- switching on everything the driver
// reports can cost performance.
static void gpu_features_enable(gpu_features_t *g)
{
#define X(member, field) g->member.field = VK_TRUE;
    GPU_REQUIRED_FEATURES(X)
#undef X
}

// ---------------------------------------------------------------------------
// instance
// ---------------------------------------------------------------------------

static bool gpu_want_validation(void)
{
    const char *env = getenv("PT_VALIDATION");
    if (env) {
        return !(env[0] == '0' && env[1] == '\0');
    }
#ifdef PT_ENABLE_VALIDATION
    return true;
#else
    return false;
#endif
}

static void gpu_create_instance(gpu_device_t *gpu)
{
    if (vkEnumerateInstanceVersion == NULL) {
        gpu_fatal("volkInitialize() must be called before gpu_device_init()");
    }

    uint32_t instance_version = 0;
    VK_CHECK(vkEnumerateInstanceVersion(&instance_version));
    if (instance_version < PT_API_VERSION) {
        gpu_fatal("Vulkan loader reports %u.%u, but 1.4 is required",
                  VK_API_VERSION_MAJOR(instance_version), VK_API_VERSION_MINOR(instance_version));
    }

    // Available layers and extensions, queried once and reused below.
    uint32_t layer_count = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&layer_count, NULL));
    VkLayerProperties *layers = gpu_alloc(layer_count * sizeof(*layers));
    VK_CHECK(vkEnumerateInstanceLayerProperties(&layer_count, layers));

    uint32_t available_count = 0;
    VK_CHECK(vkEnumerateInstanceExtensionProperties(NULL, &available_count, NULL));
    VkExtensionProperties *available = gpu_alloc(available_count * sizeof(*available));
    VK_CHECK(vkEnumerateInstanceExtensionProperties(NULL, &available_count, available));

    gpu->validation_enabled = gpu_want_validation();
    if (gpu->validation_enabled && !has_layer(layers, layer_count, PT_VALIDATION_LAYER)) {
        fprintf(stderr,
                "gpu: warning: %s not installed, continuing without validation "
                "(pacman -S vulkan-validation-layers)\n",
                PT_VALIDATION_LAYER);
        gpu->validation_enabled = false;
    }
    if (gpu->validation_enabled &&
        !has_extension(available, available_count, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        fprintf(stderr, "gpu: warning: %s unavailable, continuing without validation\n",
                VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        gpu->validation_enabled = false;
    }

    // GLFW tells us which surface extensions this platform needs (Wayland or X11).
    uint32_t glfw_count = 0;
    const char **glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_count);
    if (!glfw_extensions) {
        gpu_fatal("GLFW could not report the required Vulkan instance extensions");
    }

    const char *extensions[8];
    uint32_t extension_count = 0;
    for (uint32_t i = 0; i < glfw_count; ++i) {
        if (extension_count >= PT_COUNT(extensions)) {
            gpu_fatal("too many instance extensions requested");
        }
        if (!has_extension(available, available_count, glfw_extensions[i])) {
            gpu_fatal("instance extension %s required by GLFW is not available",
                      glfw_extensions[i]);
        }
        extensions[extension_count++] = glfw_extensions[i];
    }
    if (gpu->validation_enabled) {
        extensions[extension_count++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    }

    VkDebugUtilsMessengerCreateInfoEXT debug_info = gpu_debug_messenger_info();

    // Chained into the instance so that instance creation and destruction are themselves
    // covered by the messenger, before/after the real one exists.
    VkValidationFeatureEnableEXT validation_enables[] = {
        VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT,
    };
    VkValidationFeaturesEXT validation_features = {
        .sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
        .pNext = &debug_info,
        .enabledValidationFeatureCount = PT_COUNT(validation_enables),
        .pEnabledValidationFeatures = validation_enables,
    };

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Pathtracer",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "Pathtracer",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = PT_API_VERSION,
    };

    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = gpu->validation_enabled ? (const void *)&validation_features : NULL,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = gpu->validation_enabled ? 1u : 0u,
        .ppEnabledLayerNames = gpu->validation_enabled ? &PT_VALIDATION_LAYER : NULL,
        .enabledExtensionCount = extension_count,
        .ppEnabledExtensionNames = extensions,
    };

    VK_CHECK(vkCreateInstance(&instance_info, NULL, &gpu->instance));
    volkLoadInstanceOnly(gpu->instance);

    if (gpu->validation_enabled) {
        VK_CHECK(vkCreateDebugUtilsMessengerEXT(gpu->instance, &debug_info, NULL,
                                                &gpu->debug_messenger));
    }

    free(layers);
    free(available);
}

// ---------------------------------------------------------------------------
// queue families
// ---------------------------------------------------------------------------

typedef struct gpu_queue_families_t {
    uint32_t universal;
    uint32_t compute;
    uint32_t transfer;
} gpu_queue_families_t;

static uint32_t popcount32(uint32_t v)
{
    uint32_t n = 0;
    while (v) {
        v &= v - 1;
        ++n;
    }
    return n;
}

// Picks a universal (graphics+compute+present) family plus, where the device exposes
// them, a dedicated async-compute family and a dedicated DMA family. Roles without a
// dedicated family fall back to the universal one.
static bool gpu_select_queue_families(VkInstance instance, VkPhysicalDevice physical_device,
                                      gpu_queue_families_t *out)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, NULL);
    VkQueueFamilyProperties *families = gpu_alloc(count * sizeof(*families));
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, families);

    const uint32_t none = UINT32_MAX;
    uint32_t universal = none;
    uint32_t compute = none;
    uint32_t transfer = none;
    uint32_t compute_extra_bits = UINT32_MAX;

    for (uint32_t i = 0; i < count; ++i) {
        const VkQueueFlags flags = families[i].queueFlags;
        if (families[i].queueCount == 0) {
            continue;
        }

        const bool graphics = (flags & VK_QUEUE_GRAPHICS_BIT) != 0;
        const bool computes = (flags & VK_QUEUE_COMPUTE_BIT) != 0;
        const bool transfers = (flags & VK_QUEUE_TRANSFER_BIT) != 0;

        // glfwGetPhysicalDevicePresentationSupport answers this without needing a
        // VkSurfaceKHR, so no window/surface has to exist before device creation.
        if (universal == none && graphics && computes &&
            glfwGetPhysicalDevicePresentationSupport(instance, physical_device, i) == GLFW_TRUE) {
            universal = i;
            continue;
        }

        // Async compute: compute without graphics, preferring the most specialised family.
        if (computes && !graphics) {
            const uint32_t extra = popcount32(flags & ~(VkQueueFlags)VK_QUEUE_COMPUTE_BIT);
            if (compute == none || extra < compute_extra_bits) {
                compute = i;
                compute_extra_bits = extra;
            }
        }

        // Dedicated DMA: transfer only. Excluding the video and optical-flow bits keeps
        // the NVIDIA decode/encode/optical-flow families out of the running.
        const VkQueueFlags disqualifying =
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_VIDEO_DECODE_BIT_KHR |
            VK_QUEUE_VIDEO_ENCODE_BIT_KHR | VK_QUEUE_OPTICAL_FLOW_BIT_NV;
        if (transfer == none && transfers && (flags & disqualifying) == 0) {
            transfer = i;
        }
    }

    free(families);

    if (universal == none) {
        return false;
    }

    out->universal = universal;
    out->compute = (compute != none) ? compute : universal;
    out->transfer = (transfer != none) ? transfer : universal;
    return true;
}

// ---------------------------------------------------------------------------
// physical device
// ---------------------------------------------------------------------------

// Returns true if the device meets every requirement, otherwise fills `reason`.
static bool gpu_device_suitable(VkInstance instance, VkPhysicalDevice physical_device,
                                const VkPhysicalDeviceProperties *props,
                                gpu_queue_families_t *families, char *reason, size_t reason_size)
{
    if (props->apiVersion < PT_API_VERSION) {
        snprintf(reason, reason_size, "reports Vulkan %u.%u, needs 1.4",
                 VK_API_VERSION_MAJOR(props->apiVersion), VK_API_VERSION_MINOR(props->apiVersion));
        return false;
    }

    uint32_t available_count = 0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(physical_device, NULL, &available_count, NULL));
    VkExtensionProperties *available = gpu_alloc(available_count * sizeof(*available));
    VK_CHECK(vkEnumerateDeviceExtensionProperties(physical_device, NULL, &available_count,
                                                  available));

    for (uint32_t i = 0; i < PT_COUNT(PT_DEVICE_EXTENSIONS); ++i) {
        if (!has_extension(available, available_count, PT_DEVICE_EXTENSIONS[i])) {
            snprintf(reason, reason_size, "missing extension %s", PT_DEVICE_EXTENSIONS[i]);
            free(available);
            return false;
        }
    }
    free(available);

    gpu_features_t features;
    gpu_features_chain(&features);
    vkGetPhysicalDeviceFeatures2(physical_device, &features.base);

    const char *missing[4];
    const uint32_t missing_count = gpu_features_missing(&features, missing, PT_COUNT(missing));
    if (missing_count > 0) {
        int written = snprintf(reason, reason_size, "missing %u feature(s):", missing_count);
        for (uint32_t i = 0; i < missing_count && i < PT_COUNT(missing); ++i) {
            if (written < 0 || (size_t)written >= reason_size) {
                break;
            }
            written += snprintf(reason + written, reason_size - (size_t)written, " %s",
                                missing[i]);
        }
        return false;
    }

    if (!gpu_select_queue_families(instance, physical_device, families)) {
        snprintf(reason, reason_size, "no graphics+compute queue family with present support");
        return false;
    }

    return true;
}

static uint64_t gpu_device_local_memory(VkPhysicalDevice physical_device)
{
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory);

    uint64_t total = 0;
    for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
        if (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total += memory.memoryHeaps[i].size;
        }
    }
    return total;
}

static uint64_t gpu_device_score(VkPhysicalDeviceType type, uint64_t device_local)
{
    uint64_t score;
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: score = 1000; break;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = 100; break;
    default: score = 10; break;
    }
    // Tie-break on VRAM, scaled so it can never outrank the device type.
    return score * (1ull << 20) + (device_local >> 20);
}

static void gpu_pick_physical_device(gpu_device_t *gpu)
{
    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(gpu->instance, &count, NULL));
    if (count == 0) {
        gpu_fatal("no Vulkan physical devices found");
    }
    VkPhysicalDevice *devices = gpu_alloc(count * sizeof(*devices));
    VK_CHECK(vkEnumeratePhysicalDevices(gpu->instance, &count, devices));

    // PT_GPU_INDEX pins a specific device, for multi-GPU debugging.
    uint32_t forced_index = UINT32_MAX;
    const char *env = getenv("PT_GPU_INDEX");
    if (env && env[0] != '\0') {
        char *end = NULL;
        const unsigned long parsed = strtoul(env, &end, 10);
        if (*end != '\0' || parsed >= count) {
            gpu_fatal("PT_GPU_INDEX=%s is not a valid device index (%u device(s) present)", env,
                      count);
        }
        forced_index = (uint32_t)parsed;
    }

    const size_t reason_size = 256;
    char *reasons = gpu_alloc(count * reason_size);
    VkPhysicalDevice best = VK_NULL_HANDLE;
    gpu_queue_families_t best_families = {0};
    VkPhysicalDeviceProperties best_props = {0};
    uint64_t best_score = 0;

    for (uint32_t i = 0; i < count; ++i) {
        char *reason = reasons + i * reason_size;
        reason[0] = '\0';

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);

        if (forced_index != UINT32_MAX && i != forced_index) {
            snprintf(reason, reason_size, "not selected by PT_GPU_INDEX");
            continue;
        }

        gpu_queue_families_t families;
        if (!gpu_device_suitable(gpu->instance, devices[i], &props, &families, reason,
                                 reason_size)) {
            continue;
        }

        const uint64_t score = gpu_device_score(props.deviceType, gpu_device_local_memory(devices[i]));
        if (best == VK_NULL_HANDLE || score > best_score) {
            best = devices[i];
            best_score = score;
            best_families = families;
            best_props = props;
        }
    }

    if (best == VK_NULL_HANDLE) {
        fprintf(stderr, "gpu: no suitable device among %u:\n", count);
        for (uint32_t i = 0; i < count; ++i) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devices[i], &props);
            fprintf(stderr, "  [%u] %s: %s\n", i, props.deviceName, reasons + i * reason_size);
        }
        gpu_fatal("a Vulkan 1.4 GPU with ray tracing support is required");
    }

    gpu->physical_device = best;
    gpu->properties = best_props;
    gpu->universal_family = best_families.universal;
    gpu->compute_family = best_families.compute;
    gpu->transfer_family = best_families.transfer;

    vkGetPhysicalDeviceMemoryProperties(best, &gpu->memory_properties);

    // Handle size and alignment here are what the shader binding table will need later.
    gpu->accel_properties = (VkPhysicalDeviceAccelerationStructurePropertiesKHR){
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
    };
    gpu->rt_pipeline_properties = (VkPhysicalDeviceRayTracingPipelinePropertiesKHR){
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
        .pNext = &gpu->accel_properties,
    };
    VkPhysicalDeviceProperties2 properties2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &gpu->rt_pipeline_properties,
    };
    vkGetPhysicalDeviceProperties2(best, &properties2);

    free(reasons);
    free(devices);
}

// ---------------------------------------------------------------------------
// logical device
// ---------------------------------------------------------------------------

static void gpu_create_device(gpu_device_t *gpu)
{
    // vkCreateDevice rejects duplicate family indices, and the three roles collapse onto
    // the universal family whenever the device has no dedicated one.
    const uint32_t roles[] = {gpu->universal_family, gpu->compute_family, gpu->transfer_family};
    uint32_t unique[PT_COUNT(roles)];
    uint32_t unique_count = 0;
    for (uint32_t i = 0; i < PT_COUNT(roles); ++i) {
        bool seen = false;
        for (uint32_t j = 0; j < unique_count; ++j) {
            seen = seen || unique[j] == roles[i];
        }
        if (!seen) {
            unique[unique_count++] = roles[i];
        }
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_infos[PT_COUNT(roles)];
    for (uint32_t i = 0; i < unique_count; ++i) {
        queue_infos[i] = (VkDeviceQueueCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = unique[i],
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };
    }

    gpu_features_t features;
    gpu_features_chain(&features);
    gpu_features_enable(&features);

    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features.base, // features go through pNext, so pEnabledFeatures stays NULL
        .queueCreateInfoCount = unique_count,
        .pQueueCreateInfos = queue_infos,
        .enabledExtensionCount = PT_COUNT(PT_DEVICE_EXTENSIONS),
        .ppEnabledExtensionNames = PT_DEVICE_EXTENSIONS,
    };

    VK_CHECK(vkCreateDevice(gpu->physical_device, &device_info, NULL, &gpu->device));
    volkLoadDevice(gpu->device);

    vkGetDeviceQueue(gpu->device, gpu->universal_family, 0, &gpu->universal_queue);
    vkGetDeviceQueue(gpu->device, gpu->compute_family, 0, &gpu->compute_queue);
    vkGetDeviceQueue(gpu->device, gpu->transfer_family, 0, &gpu->transfer_queue);
}

// ---------------------------------------------------------------------------
// report
// ---------------------------------------------------------------------------

static const char *gpu_device_type_string(VkPhysicalDeviceType type)
{
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
    default: return "other";
    }
}

static void gpu_print_report(const gpu_device_t *gpu)
{
    VkPhysicalDeviceDriverProperties driver = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 properties2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &driver,
    };
    vkGetPhysicalDeviceProperties2(gpu->physical_device, &properties2);

    const uint32_t api = gpu->properties.apiVersion;

    printf("GPU: %s (%s)\n", gpu->properties.deviceName,
           gpu_device_type_string(gpu->properties.deviceType));
    printf("  Vulkan     %u.%u.%u\n", VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api),
           VK_API_VERSION_PATCH(api));
    printf("  Driver     %s %s\n", driver.driverName, driver.driverInfo);
    printf("  VRAM       %llu MiB\n",
           (unsigned long long)(gpu_device_local_memory(gpu->physical_device) >> 20));
    printf("  Queues     universal=%u compute=%u transfer=%u\n", gpu->universal_family,
           gpu->compute_family, gpu->transfer_family);
    printf("  Validation %s\n", gpu->validation_enabled ? "enabled" : "disabled");
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

void gpu_device_init(gpu_device_t *device)
{
    memset(device, 0, sizeof(*device));

    gpu_create_instance(device);
    gpu_pick_physical_device(device);
    gpu_create_device(device);
    gpu_print_report(device);
}

void gpu_device_free(gpu_device_t *device)
{
    if (device->device) {
        vkDestroyDevice(device->device, NULL);
    }
    if (device->debug_messenger) {
        vkDestroyDebugUtilsMessengerEXT(device->instance, device->debug_messenger, NULL);
    }
    if (device->instance) {
        vkDestroyInstance(device->instance, NULL);
    }
    memset(device, 0, sizeof(*device));
}
