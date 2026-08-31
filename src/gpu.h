#pragma once

#define VK_NO_PROTOTYPES
#include <volk.h>

#include <stdbool.h>

#define GPU_FRAMES_IN_FLIGHT 2
#define GPU_MAX_SWAPCHAIN_IMAGES 8

// Forward declared so this header stays free of glfw3.h.
typedef struct GLFWwindow GLFWwindow;

// ---------------------------------------------------------------------------
// device
// ---------------------------------------------------------------------------

typedef struct gpu_device_t {
    VkInstance instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    VkPhysicalDevice physical_device;
    VkDevice device;

    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceMemoryProperties memory_properties;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_pipeline_properties;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accel_properties;

    uint32_t universal_family; // graphics + compute + transfer + present
    uint32_t compute_family;   // async compute, == universal_family if none dedicated
    uint32_t transfer_family;  // DMA,           == universal_family if none dedicated
    VkQueue universal_queue;
    VkQueue compute_queue;
    VkQueue transfer_queue;

    bool validation_enabled;
} gpu_device_t;

void gpu_device_init(gpu_device_t *device);
void gpu_device_free(gpu_device_t *device);

// ---------------------------------------------------------------------------
// swapchain
// ---------------------------------------------------------------------------

typedef struct gpu_swapchain_t {
    gpu_device_t *device; // not owned
    GLFWwindow *window;   // not owned

    VkSurfaceKHR surface;
    VkSwapchainKHR handle;
    VkFormat format;
    VkColorSpaceKHR color_space;
    VkPresentModeKHR present_mode;
    VkExtent2D extent;
    // Framebuffer size this swapchain was built for. Compared against the live window
    // size to detect resizes. Not the same as `extent`: on X11 `extent` comes from
    // VkSurfaceCapabilitiesKHR::currentExtent, which can differ transiently and would
    // otherwise force a recreate every frame.
    VkExtent2D window_extent;

    uint32_t image_count;
    VkImage images[GPU_MAX_SWAPCHAIN_IMAGES];
    VkImageView views[GPU_MAX_SWAPCHAIN_IMAGES];
    // Per image, NOT per frame-in-flight: a present can still be pending on this
    // semaphore by the time the frame slot comes round again.
    VkSemaphore release[GPU_MAX_SWAPCHAIN_IMAGES];

    bool out_of_date;
} gpu_swapchain_t;

void gpu_swapchain_init(gpu_swapchain_t *swapchain, gpu_device_t *device, GLFWwindow *window);
void gpu_swapchain_recreate(gpu_swapchain_t *swapchain);
void gpu_swapchain_free(gpu_swapchain_t *swapchain);

// ---------------------------------------------------------------------------
// frames
// ---------------------------------------------------------------------------

typedef struct gpu_frame_t {
    VkCommandPool pool;
    VkCommandBuffer cmd;
    VkSemaphore acquire; // binary; the swapchain cannot signal a timeline semaphore
    // Index of this slot in the ring, fixed for the life of the frames object. Callers that
    // keep their own per-frame-in-flight resources index them with this.
    uint32_t slot;

    // Filled in by gpu_frame_begin.
    VkImage image;
    VkImageView view;
    VkExtent2D extent;
    uint32_t image_index;
} gpu_frame_t;

typedef struct gpu_frames_t {
    gpu_device_t *device; // not owned
    gpu_frame_t frame[GPU_FRAMES_IN_FLIGHT];
    VkSemaphore timeline; // paces the CPU against the GPU, replacing per-frame fences
    uint64_t submitted;   // frames submitted == last value signalled on the timeline
} gpu_frames_t;

void gpu_frames_init(gpu_frames_t *frames, gpu_device_t *device);
void gpu_frames_free(gpu_frames_t *frames);

// Returns NULL when there is nothing to draw this tick: the window is minimised, or the
// swapchain went out of date and was recreated. The returned pointer is owned by `frames`
// and stays valid until the matching gpu_frame_end.
gpu_frame_t *gpu_frame_begin(gpu_frames_t *frames, gpu_swapchain_t *swapchain);
void gpu_frame_end(gpu_frames_t *frames, gpu_swapchain_t *swapchain, gpu_frame_t *frame);

// ---------------------------------------------------------------------------
// memory
// ---------------------------------------------------------------------------

typedef enum gpu_memory_kind_t {
    GPU_MEMORY_DEVICE,   // device-local, not host visible
    GPU_MEMORY_UPLOAD,   // host visible + coherent, persistently mapped
    GPU_MEMORY_READBACK, // host visible + coherent + cached, persistently mapped
} gpu_memory_kind_t;

typedef struct gpu_buffer_t {
    VkBuffer handle;
    VkDeviceMemory memory;
    VkDeviceAddress address; // 0 unless created with SHADER_DEVICE_ADDRESS usage
    VkDeviceSize size;
    void *mapped; // NULL for GPU_MEMORY_DEVICE
} gpu_buffer_t;

typedef struct gpu_image_t {
    VkImage handle;
    VkDeviceMemory memory;
    VkImageView view;
    VkFormat format;
    VkExtent2D extent;
} gpu_image_t;

// One vkAllocateMemory per resource for now. Suballocation, when it is needed, goes
// behind this API without touching any call site.
gpu_buffer_t gpu_buffer_create(gpu_device_t *device, VkDeviceSize size,
                               VkBufferUsageFlags usage, gpu_memory_kind_t kind);
void gpu_buffer_destroy(gpu_device_t *device, gpu_buffer_t *buffer);

gpu_image_t gpu_image_create(gpu_device_t *device, VkExtent2D extent, VkFormat format,
                             VkImageUsageFlags usage);
void gpu_image_destroy(gpu_device_t *device, gpu_image_t *image);

// Synchronous one-shot command submission, used by uploads and acceleration structure
// builds. Runs on the universal queue; moving transfers to gpu_device_t::transfer_family
// is a later change contained to this object.
typedef struct gpu_uploader_t {
    gpu_device_t *device; // not owned
    VkCommandPool pool;
    VkCommandBuffer cmd;
    VkFence fence;
} gpu_uploader_t;

void gpu_uploader_init(gpu_uploader_t *uploader, gpu_device_t *device);
void gpu_uploader_free(gpu_uploader_t *uploader);

VkCommandBuffer gpu_upload_begin(gpu_uploader_t *uploader);
void gpu_upload_end(gpu_uploader_t *uploader); // submits and waits

// Stages through a temporary host-visible buffer.
void gpu_buffer_upload(gpu_uploader_t *uploader, gpu_buffer_t *dst, const void *data,
                       VkDeviceSize size);

// Fills the whole of `dst` from tightly packed `data` and leaves it in GENERAL, ready to be
// sampled. Only the single mip level and array layer gpu_image_create makes.
void gpu_image_upload(gpu_uploader_t *uploader, gpu_image_t *dst, const void *data,
                      VkDeviceSize size);

// ---------------------------------------------------------------------------
// shader modules
// ---------------------------------------------------------------------------

// Reads a SPIR-V file and creates a shader module from it. Returns VK_NULL_HANDLE and logs
// rather than aborting, so a missing or broken .spv cannot take the process down during a
// hot reload.
VkShaderModule gpu_shader_module_load(gpu_device_t *device, const char *path);

// ---------------------------------------------------------------------------
// acceleration structures
// ---------------------------------------------------------------------------

typedef struct gpu_accel_t {
    VkAccelerationStructureKHR handle;
    gpu_buffer_t buffer;
    VkDeviceAddress address;
} gpu_accel_t;

// The generic builder every acceleration structure goes through; the wrappers below are
// thin conveniences over it.
gpu_accel_t gpu_accel_build(gpu_device_t *device, gpu_uploader_t *uploader,
                            VkAccelerationStructureTypeKHR type,
                            const VkAccelerationStructureGeometryKHR *geometries,
                            const uint32_t *primitive_counts, uint32_t geometry_count);

gpu_accel_t gpu_blas_build_triangles(gpu_device_t *device, gpu_uploader_t *uploader,
                                     VkDeviceAddress vertices, uint32_t vertex_count,
                                     VkDeviceSize vertex_stride, VkDeviceAddress indices,
                                     uint32_t index_count);

// Procedural geometry: `count` VkAabbPositionsKHR, each intersected by an intersection
// shader chosen through the hit group the instance points at.
gpu_accel_t gpu_blas_build_aabbs(gpu_device_t *device, gpu_uploader_t *uploader,
                                 VkDeviceAddress aabbs, uint32_t count);

// Uploads `instances` into a build-input buffer and builds a top level structure over it.
gpu_accel_t gpu_tlas_build(gpu_device_t *device, gpu_uploader_t *uploader,
                           const VkAccelerationStructureInstanceKHR *instances,
                           uint32_t instance_count);

void gpu_accel_destroy(gpu_device_t *device, gpu_accel_t *accel);

// ---------------------------------------------------------------------------
// ray tracing pipeline
// ---------------------------------------------------------------------------

typedef struct gpu_sbt_t {
    gpu_buffer_t buffer;
    VkStridedDeviceAddressRegionKHR raygen;
    VkStridedDeviceAddressRegionKHR miss;
    VkStridedDeviceAddressRegionKHR hit;
    VkStridedDeviceAddressRegionKHR callable;
} gpu_sbt_t;

typedef struct gpu_rt_pipeline_t {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    gpu_sbt_t sbt;
} gpu_rt_pipeline_t;

// One hit group, i.e. one shader binding table hit record. Its index is what a TLAS
// instance selects through instanceShaderBindingTableRecordOffset.
typedef struct gpu_hit_group_t {
    const char *closest_hit;  // required
    const char *any_hit;      // NULL for none
    const char *intersection; // NULL selects a triangles group, otherwise a procedural one
} gpu_hit_group_t;

// Every stage comes from one multi-entry-point SPIR-V module, selected by entry point
// name. This is why the shaders must be compiled with -fvk-use-entrypoint-name. The same
// entry point may appear in several hit groups; it simply becomes several stages.
typedef struct gpu_rt_pipeline_desc_t {
    const char *spirv_path;
    const char *raygen_entry;
    const char *const *miss_entries;
    uint32_t miss_count;
    const gpu_hit_group_t *hit_groups;
    uint32_t hit_group_count;
    VkDescriptorSetLayout set_layout;
    uint32_t push_constant_size;
    uint32_t max_recursion;
} gpu_rt_pipeline_desc_t;

// The stages the push constant range covers. vkCmdPushConstants has to name exactly this
// set, not a subset of it, so both sides take it from here.
#define GPU_RT_PUSH_CONSTANT_STAGES                                                        \
    (VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |                       \
     VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |               \
     VK_SHADER_STAGE_INTERSECTION_BIT_KHR)

// Returns false rather than aborting: hot reload has to survive a broken shader.
bool gpu_rt_pipeline_create(gpu_device_t *device, gpu_uploader_t *uploader,
                            const gpu_rt_pipeline_desc_t *desc, gpu_rt_pipeline_t *out);
void gpu_rt_pipeline_destroy(gpu_device_t *device, gpu_rt_pipeline_t *pipeline);

// ---------------------------------------------------------------------------
// command helpers
// ---------------------------------------------------------------------------

// Maps 1:1 onto VkImageMemoryBarrier2. With VK_KHR_unified_image_layouts enabled most
// images can stay in VK_IMAGE_LAYOUT_GENERAL, so old_layout == new_layout is the common
// case and this expresses a pure memory/execution dependency.
typedef struct gpu_image_barrier_t {
    VkImage image;
    VkImageLayout old_layout;
    VkImageLayout new_layout;
    VkPipelineStageFlags2 src_stage;
    VkPipelineStageFlags2 dst_stage;
    VkAccessFlags2 src_access;
    VkAccessFlags2 dst_access;
} gpu_image_barrier_t;

void gpu_cmd_image_barrier(VkCommandBuffer cmd, const gpu_image_barrier_t *barrier);

// Passing clear == NULL loads the existing contents instead of clearing.
void gpu_cmd_begin_rendering(VkCommandBuffer cmd, VkImageView view, VkExtent2D extent,
                             const VkClearColorValue *clear);
void gpu_cmd_end_rendering(VkCommandBuffer cmd);
