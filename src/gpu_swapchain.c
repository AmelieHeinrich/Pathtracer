#include "gpu_internal.h"

#include <GLFW/glfw3.h>

#include <string.h>

// Chosen so the pathtracer can later blit a tonemapped image straight in. Deliberately
// no STORAGE bit: sRGB formats do not support VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT, so a
// compute pass cannot write the swapchain image directly.
#define PT_SWAPCHAIN_USAGE                                                               \
    (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)

static uint32_t clamp_u32(uint32_t value, uint32_t low, uint32_t high)
{
    return value < low ? low : (value > high ? high : value);
}

static VkSurfaceFormatKHR choose_surface_format(VkPhysicalDevice physical_device,
                                                VkSurfaceKHR surface)
{
    uint32_t count = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &count, NULL));
    if (count == 0) {
        gpu_fatal("surface reports no formats");
    }
    VkSurfaceFormatKHR *formats = gpu_alloc(count * sizeof(*formats));
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &count, formats));

    // An sRGB format lets the hardware do the linear->sRGB encode on write, which is what
    // we want once a pathtracer starts handing over tonemapped linear radiance.
    const VkFormat preferred[] = {VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB};

    VkSurfaceFormatKHR chosen = formats[0];
    for (uint32_t i = 0; i < PT_COUNT(preferred); ++i) {
        bool found = false;
        for (uint32_t j = 0; j < count && !found; ++j) {
            if (formats[j].format == preferred[i] &&
                formats[j].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                chosen = formats[j];
                found = true;
            }
        }
        if (found) {
            break;
        }
    }

    free(formats);
    return chosen;
}

static VkPresentModeKHR choose_present_mode(VkPhysicalDevice physical_device,
                                            VkSurfaceKHR surface)
{
    uint32_t count = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &count, NULL));
    VkPresentModeKHR *modes = gpu_alloc(count * sizeof(*modes));
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &count, modes));

    // FIFO is the only mode Vulkan guarantees; MAILBOX just avoids blocking on vblank.
    VkPresentModeKHR chosen = VK_PRESENT_MODE_FIFO_KHR;
    for (uint32_t i = 0; i < count; ++i) {
        if (modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
            chosen = VK_PRESENT_MODE_MAILBOX_KHR;
            break;
        }
    }

    free(modes);
    return chosen;
}

// Destroys everything tied to a particular VkSwapchainKHR, leaving the surface alone.
static void destroy_swapchain_resources(gpu_swapchain_t *swapchain)
{
    VkDevice device = swapchain->device->device;
    for (uint32_t i = 0; i < swapchain->image_count; ++i) {
        vkDestroyImageView(device, swapchain->views[i], NULL);
        vkDestroySemaphore(device, swapchain->release[i], NULL);
    }
    swapchain->image_count = 0;
}

// Builds a new VkSwapchainKHR, retiring whatever `swapchain->handle` currently holds.
// A zero-sized framebuffer (minimised window) leaves handle == VK_NULL_HANDLE.
static void build_swapchain(gpu_swapchain_t *swapchain)
{
    gpu_device_t *gpu = swapchain->device;
    VkPhysicalDevice physical_device = gpu->physical_device;
    VkDevice device = gpu->device;

    VkSurfaceCapabilitiesKHR caps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, swapchain->surface,
                                                       &caps));

    int window_width = 0;
    int window_height = 0;
    glfwGetFramebufferSize(swapchain->window, &window_width, &window_height);
    const VkExtent2D window_extent = {(uint32_t)window_width, (uint32_t)window_height};

    // Wayland reports currentExtent as UINT32_MAX and expects us to pick the size.
    VkExtent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) {
        extent.width = clamp_u32(window_extent.width, caps.minImageExtent.width,
                                 caps.maxImageExtent.width);
        extent.height = clamp_u32(window_extent.height, caps.minImageExtent.height,
                                  caps.maxImageExtent.height);
    }
    swapchain->window_extent = window_extent;

    VkSwapchainKHR old_handle = swapchain->handle;

    if (extent.width == 0 || extent.height == 0) {
        // Minimised: retire the old swapchain and leave nothing presentable behind.
        destroy_swapchain_resources(swapchain);
        if (old_handle) {
            vkDestroySwapchainKHR(device, old_handle, NULL);
        }
        swapchain->handle = VK_NULL_HANDLE;
        swapchain->extent = extent;
        return;
    }

    uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
        image_count = caps.maxImageCount;
    }
    if (image_count > GPU_MAX_SWAPCHAIN_IMAGES) {
        image_count = GPU_MAX_SWAPCHAIN_IMAGES;
    }

    const VkImageUsageFlags usage = PT_SWAPCHAIN_USAGE & caps.supportedUsageFlags;
    if (!(usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) {
        gpu_fatal("surface does not support colour attachment usage");
    }

    const VkSurfaceFormatKHR format = choose_surface_format(physical_device, swapchain->surface);

    VkSwapchainCreateInfoKHR info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = swapchain->surface,
        .minImageCount = image_count,
        .imageFormat = format.format,
        .imageColorSpace = format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = usage,
        // A single universal queue both renders and presents, so no sharing is needed.
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = caps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = choose_present_mode(physical_device, swapchain->surface),
        .clipped = VK_TRUE,
        .oldSwapchain = old_handle,
    };

    VK_CHECK(vkCreateSwapchainKHR(device, &info, NULL, &swapchain->handle));

    // Only safe after the new swapchain has been created from it.
    destroy_swapchain_resources(swapchain);
    if (old_handle) {
        vkDestroySwapchainKHR(device, old_handle, NULL);
    }

    swapchain->format = format.format;
    swapchain->color_space = format.colorSpace;
    swapchain->present_mode = info.presentMode;
    swapchain->extent = extent;
    swapchain->out_of_date = false;

    uint32_t actual_count = 0;
    VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain->handle, &actual_count, NULL));
    if (actual_count > GPU_MAX_SWAPCHAIN_IMAGES) {
        gpu_fatal("swapchain returned %u images, GPU_MAX_SWAPCHAIN_IMAGES is %u", actual_count,
                  GPU_MAX_SWAPCHAIN_IMAGES);
    }
    VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain->handle, &actual_count,
                                     swapchain->images));
    swapchain->image_count = actual_count;

    for (uint32_t i = 0; i < actual_count; ++i) {
        VkImageViewCreateInfo view_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = swapchain->images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = swapchain->format,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .layerCount = 1,
            },
        };
        VK_CHECK(vkCreateImageView(device, &view_info, NULL, &swapchain->views[i]));

        const VkSemaphoreCreateInfo semaphore_info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };
        VK_CHECK(vkCreateSemaphore(device, &semaphore_info, NULL, &swapchain->release[i]));
    }
}

void gpu_swapchain_init(gpu_swapchain_t *swapchain, gpu_device_t *device, GLFWwindow *window)
{
    memset(swapchain, 0, sizeof(*swapchain));
    swapchain->device = device;
    swapchain->window = window;

    VK_CHECK(glfwCreateWindowSurface(device->instance, window, NULL, &swapchain->surface));

    // Device selection used GLFW's platform-level presentation check; this is the
    // surface-level one, which is the condition vkQueuePresentKHR actually requires.
    VkBool32 supported = VK_FALSE;
    VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(device->physical_device,
                                                  device->universal_family,
                                                  swapchain->surface, &supported));
    if (!supported) {
        gpu_fatal("queue family %u cannot present to this surface", device->universal_family);
    }

    build_swapchain(swapchain);

    printf("Swapchain: %ux%u, %u images, %s\n", swapchain->extent.width, swapchain->extent.height,
           swapchain->image_count,
           swapchain->present_mode == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : "fifo");
    fflush(stdout);
}

void gpu_swapchain_recreate(gpu_swapchain_t *swapchain)
{
    // The old images and semaphores may still be referenced by in-flight work.
    VK_CHECK(vkDeviceWaitIdle(swapchain->device->device));
    build_swapchain(swapchain);
}

void gpu_swapchain_free(gpu_swapchain_t *swapchain)
{
    if (!swapchain->device) {
        return;
    }
    VkDevice device = swapchain->device->device;

    destroy_swapchain_resources(swapchain);
    if (swapchain->handle) {
        vkDestroySwapchainKHR(device, swapchain->handle, NULL);
    }
    if (swapchain->surface) {
        vkDestroySurfaceKHR(swapchain->device->instance, swapchain->surface, NULL);
    }
    memset(swapchain, 0, sizeof(*swapchain));
}
