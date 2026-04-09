/*
 * test_vulkan.c — RGPU Vulkan ICD integration test
 *
 * Tests: vkCreateInstance, vkEnumeratePhysicalDevices,
 *        vkGetPhysicalDeviceProperties, vkGetPhysicalDeviceMemoryProperties,
 *        vkCreateDevice, vkCreateBuffer, vkAllocateMemory,
 *        vkBindBufferMemory, cleanup
 *
 * Compile: gcc -o test_vulkan test_vulkan.c -lvulkan
 * Run with: VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/rgpu_icd.json ./test_vulkan
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

static int pass = 0, fail = 0;

#define CHECK_VK(name, expr) do { \
    VkResult r = (expr); \
    if (r == VK_SUCCESS) { printf("  PASS: %s\n", name); pass++; } \
    else { printf("  FAIL: %s (VkResult %d)\n", name, r); fail++; } \
} while(0)

int main(void) {
    printf("=== Vulkan ICD Test ===\n");

    /* 1. Create instance */
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "rgpu-test",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "rgpu",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo inst_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
    };
    VkInstance instance = VK_NULL_HANDLE;
    CHECK_VK("vkCreateInstance", vkCreateInstance(&inst_info, NULL, &instance));
    if (instance == VK_NULL_HANDLE) {
        printf("FAIL: no instance created, aborting\n");
        return 1;
    }

    /* 2. Enumerate physical devices */
    uint32_t dev_count = 0;
    CHECK_VK("vkEnumeratePhysicalDevices(count)", vkEnumeratePhysicalDevices(instance, &dev_count, NULL));
    if (dev_count == 0) {
        printf("FAIL: no physical devices\n");
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    printf("  INFO: %u physical device(s)\n", dev_count);

    VkPhysicalDevice* phys_devs = calloc(dev_count, sizeof(VkPhysicalDevice));
    CHECK_VK("vkEnumeratePhysicalDevices(list)", vkEnumeratePhysicalDevices(instance, &dev_count, phys_devs));
    VkPhysicalDevice phys = phys_devs[0];
    free(phys_devs);

    /* 3. Get properties */
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    printf("  PASS: vkGetPhysicalDeviceProperties\n"); pass++;
    printf("  INFO: device = \"%s\"\n", props.deviceName);

    /* 4. Get memory properties */
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
    printf("  PASS: vkGetPhysicalDeviceMemoryProperties\n"); pass++;
    printf("  INFO: %u memory types, %u heaps\n", mem_props.memoryTypeCount, mem_props.memoryHeapCount);

    /* 5. Create logical device */
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };
    VkDeviceCreateInfo dev_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
    };
    VkDevice device = VK_NULL_HANDLE;
    CHECK_VK("vkCreateDevice", vkCreateDevice(phys, &dev_info, NULL, &device));
    if (device == VK_NULL_HANDLE) {
        printf("FAIL: no device created, aborting\n");
        vkDestroyInstance(instance, NULL);
        return 1;
    }

    /* 6. Create a buffer */
    VkBufferCreateInfo buf_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = 1024,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer buffer = VK_NULL_HANDLE;
    CHECK_VK("vkCreateBuffer", vkCreateBuffer(device, &buf_info, NULL, &buffer));

    /* 7. Get memory requirements and allocate */
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);
    printf("  PASS: vkGetBufferMemoryRequirements\n"); pass++;

    /* Find a memory type */
    uint32_t mem_type_idx = 0;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if (mem_reqs.memoryTypeBits & (1 << i)) {
            mem_type_idx = i;
            break;
        }
    }

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = mem_type_idx,
    };
    VkDeviceMemory memory = VK_NULL_HANDLE;
    CHECK_VK("vkAllocateMemory", vkAllocateMemory(device, &alloc_info, NULL, &memory));

    /* 8. Bind buffer memory */
    CHECK_VK("vkBindBufferMemory", vkBindBufferMemory(device, buffer, memory, 0));

    /* 9. Cleanup */
    vkDestroyBuffer(device, buffer, NULL);
    printf("  PASS: vkDestroyBuffer\n"); pass++;

    vkFreeMemory(device, memory, NULL);
    printf("  PASS: vkFreeMemory\n"); pass++;

    vkDestroyDevice(device, NULL);
    printf("  PASS: vkDestroyDevice\n"); pass++;

    vkDestroyInstance(instance, NULL);
    printf("  PASS: vkDestroyInstance\n"); pass++;

    printf("=== Vulkan: %d passed, %d failed ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
