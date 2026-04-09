/*
 * test_nvml.c — RGPU NVML interpose integration test
 *
 * Tests: nvmlInit_v2, nvmlDeviceGetCount_v2, nvmlDeviceGetHandleByIndex_v2,
 *        nvmlDeviceGetName, nvmlDeviceGetTemperature, nvmlShutdown
 *
 * Run with: LD_PRELOAD=/usr/lib/rgpu/librgpu_nvml_interpose.so ./test_nvml
 */
#include <stdio.h>
#include <dlfcn.h>
#include <stdint.h>

#define NVML_SUCCESS 0
#define NVML_TEMPERATURE_GPU 0

typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;

typedef nvmlReturn_t (*pfn_nvmlInit_v2)(void);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetCount_v2)(unsigned int*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetHandleByIndex_v2)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetName)(nvmlDevice_t, char*, unsigned int);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetTemperature)(nvmlDevice_t, int, unsigned int*);
typedef nvmlReturn_t (*pfn_nvmlShutdown)(void);

static int pass = 0, fail = 0;

#define CHECK(name, expr) do { \
    nvmlReturn_t r = (expr); \
    if (r == NVML_SUCCESS) { printf("  PASS: %s\n", name); pass++; } \
    else { printf("  FAIL: %s (error %d)\n", name, r); fail++; } \
} while(0)

int main(void) {
    printf("=== NVML Interpose Test ===\n");

    void* lib = dlopen("librgpu_nvml_interpose.so", RTLD_NOW);
    if (!lib) {
        printf("FAIL: dlopen: %s\n", dlerror());
        return 1;
    }

    pfn_nvmlInit_v2 nvmlInit = (pfn_nvmlInit_v2)dlsym(lib, "nvmlInit_v2");
    pfn_nvmlDeviceGetCount_v2 nvmlDeviceGetCount = (pfn_nvmlDeviceGetCount_v2)dlsym(lib, "nvmlDeviceGetCount_v2");
    pfn_nvmlDeviceGetHandleByIndex_v2 nvmlDeviceGetHandleByIndex =
        (pfn_nvmlDeviceGetHandleByIndex_v2)dlsym(lib, "nvmlDeviceGetHandleByIndex_v2");
    pfn_nvmlDeviceGetName nvmlDeviceGetName = (pfn_nvmlDeviceGetName)dlsym(lib, "nvmlDeviceGetName");
    pfn_nvmlDeviceGetTemperature nvmlDeviceGetTemperature =
        (pfn_nvmlDeviceGetTemperature)dlsym(lib, "nvmlDeviceGetTemperature");
    pfn_nvmlShutdown nvmlShutdown = (pfn_nvmlShutdown)dlsym(lib, "nvmlShutdown");

    if (!nvmlInit || !nvmlDeviceGetCount || !nvmlDeviceGetHandleByIndex ||
        !nvmlDeviceGetName || !nvmlDeviceGetTemperature || !nvmlShutdown) {
        printf("FAIL: missing symbols\n");
        dlclose(lib);
        return 1;
    }

    /* 1. Init */
    CHECK("nvmlInit_v2", nvmlInit());

    /* 2. Device count */
    unsigned int count = 0;
    CHECK("nvmlDeviceGetCount_v2", nvmlDeviceGetCount(&count));
    if (count == 0) {
        printf("FAIL: no devices found\n");
        dlclose(lib);
        return 1;
    }
    printf("  INFO: %u device(s)\n", count);

    /* 3. Get handle */
    nvmlDevice_t device = NULL;
    CHECK("nvmlDeviceGetHandleByIndex_v2(0)", nvmlDeviceGetHandleByIndex(0, &device));

    /* 4. Get name */
    char name[256] = {0};
    CHECK("nvmlDeviceGetName", nvmlDeviceGetName(device, name, 256));
    printf("  INFO: device name = \"%s\"\n", name);

    /* 5. Get temperature */
    unsigned int temp = 0;
    CHECK("nvmlDeviceGetTemperature", nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp));
    printf("  INFO: temperature = %u C\n", temp);

    /* 6. Shutdown */
    CHECK("nvmlShutdown", nvmlShutdown());

    dlclose(lib);

    printf("=== NVML: %d passed, %d failed ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
