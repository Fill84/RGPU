/*
 * test_cuda.c — RGPU CUDA interpose integration test
 *
 * Tests: cuInit, cuDeviceGetCount, cuDeviceGet, cuDeviceGetName,
 *        cuCtxCreate_v2, cuMemAlloc_v2, cuMemcpyHtoD_v2, cuMemcpyDtoH_v2,
 *        cuMemFree_v2, cuCtxDestroy_v2
 *
 * Run with: LD_PRELOAD=/usr/lib/rgpu/librgpu_cuda_interpose.so ./test_cuda
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>

#define CUDA_SUCCESS 0

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef uint64_t CUdeviceptr;

/* Function pointer types */
typedef CUresult (*pfn_cuInit)(unsigned int);
typedef CUresult (*pfn_cuDeviceGetCount)(int*);
typedef CUresult (*pfn_cuDeviceGet)(CUdevice*, int);
typedef CUresult (*pfn_cuDeviceGetName)(char*, int, CUdevice);
typedef CUresult (*pfn_cuCtxCreate_v2)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*pfn_cuMemAlloc_v2)(CUdeviceptr*, size_t);
typedef CUresult (*pfn_cuMemcpyHtoD_v2)(CUdeviceptr, const void*, size_t);
typedef CUresult (*pfn_cuMemcpyDtoH_v2)(void*, CUdeviceptr, size_t);
typedef CUresult (*pfn_cuMemFree_v2)(CUdeviceptr);
typedef CUresult (*pfn_cuCtxDestroy_v2)(CUcontext);

static int pass = 0, fail = 0;

#define CHECK(name, expr) do { \
    CUresult r = (expr); \
    if (r == CUDA_SUCCESS) { printf("  PASS: %s\n", name); pass++; } \
    else { printf("  FAIL: %s (error %d)\n", name, r); fail++; } \
} while(0)

int main(void) {
    printf("=== CUDA Interpose Test ===\n");

    void* lib = dlopen("librgpu_cuda_interpose.so", RTLD_NOW);
    if (!lib) {
        printf("FAIL: dlopen: %s\n", dlerror());
        return 1;
    }

    pfn_cuInit cuInit = (pfn_cuInit)dlsym(lib, "cuInit");
    pfn_cuDeviceGetCount cuDeviceGetCount = (pfn_cuDeviceGetCount)dlsym(lib, "cuDeviceGetCount");
    pfn_cuDeviceGet cuDeviceGet = (pfn_cuDeviceGet)dlsym(lib, "cuDeviceGet");
    pfn_cuDeviceGetName cuDeviceGetName = (pfn_cuDeviceGetName)dlsym(lib, "cuDeviceGetName");
    pfn_cuCtxCreate_v2 cuCtxCreate = (pfn_cuCtxCreate_v2)dlsym(lib, "cuCtxCreate_v2");
    pfn_cuMemAlloc_v2 cuMemAlloc = (pfn_cuMemAlloc_v2)dlsym(lib, "cuMemAlloc_v2");
    pfn_cuMemcpyHtoD_v2 cuMemcpyHtoD = (pfn_cuMemcpyHtoD_v2)dlsym(lib, "cuMemcpyHtoD_v2");
    pfn_cuMemcpyDtoH_v2 cuMemcpyDtoH = (pfn_cuMemcpyDtoH_v2)dlsym(lib, "cuMemcpyDtoH_v2");
    pfn_cuMemFree_v2 cuMemFree = (pfn_cuMemFree_v2)dlsym(lib, "cuMemFree_v2");
    pfn_cuCtxDestroy_v2 cuCtxDestroy = (pfn_cuCtxDestroy_v2)dlsym(lib, "cuCtxDestroy_v2");

    if (!cuInit || !cuDeviceGetCount || !cuDeviceGet || !cuDeviceGetName ||
        !cuCtxCreate || !cuMemAlloc || !cuMemcpyHtoD || !cuMemcpyDtoH ||
        !cuMemFree || !cuCtxDestroy) {
        printf("FAIL: missing symbols\n");
        dlclose(lib);
        return 1;
    }

    /* 1. Init */
    CHECK("cuInit", cuInit(0));

    /* 2. Device enumeration */
    int count = 0;
    CHECK("cuDeviceGetCount", cuDeviceGetCount(&count));
    if (count <= 0) {
        printf("FAIL: no devices found (count=%d)\n", count);
        dlclose(lib);
        return 1;
    }
    printf("  INFO: %d device(s) found\n", count);

    /* 3. Get device 0 */
    CUdevice dev = -1;
    CHECK("cuDeviceGet(0)", cuDeviceGet(&dev, 0));

    /* 4. Get device name */
    char name[256] = {0};
    CHECK("cuDeviceGetName", cuDeviceGetName(name, 256, dev));
    printf("  INFO: device name = \"%s\"\n", name);

    /* 5. Create context */
    CUcontext ctx = NULL;
    CHECK("cuCtxCreate_v2", cuCtxCreate(&ctx, 0, dev));

    /* 6. Allocate device memory (1 KB) */
    CUdeviceptr dptr = 0;
    CHECK("cuMemAlloc_v2(1024)", cuMemAlloc(&dptr, 1024));

    /* 7. Upload test pattern */
    unsigned char host_data[1024];
    for (int i = 0; i < 1024; i++) host_data[i] = (unsigned char)(i & 0xFF);
    CHECK("cuMemcpyHtoD_v2", cuMemcpyHtoD(dptr, host_data, 1024));

    /* 8. Download and verify */
    unsigned char result[1024];
    memset(result, 0, 1024);
    CHECK("cuMemcpyDtoH_v2", cuMemcpyDtoH(result, dptr, 1024));

    if (memcmp(host_data, result, 1024) == 0) {
        printf("  PASS: data round-trip verified\n");
        pass++;
    } else {
        printf("  FAIL: data mismatch after round-trip\n");
        fail++;
    }

    /* 9. Cleanup */
    CHECK("cuMemFree_v2", cuMemFree(dptr));
    CHECK("cuCtxDestroy_v2", cuCtxDestroy(ctx));

    dlclose(lib);

    printf("=== CUDA: %d passed, %d failed ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
