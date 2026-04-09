/*
 * test_nvenc.c — RGPU NVENC interpose integration test
 *
 * Tests: NvEncodeAPIGetMaxSupportedVersion, NvEncodeAPICreateInstance
 *        Validates entry points and vtable population.
 *
 * Run with: LD_PRELOAD=/usr/lib/rgpu/librgpu_nvenc_interpose.so ./test_nvenc
 */
#include <stdio.h>
#include <dlfcn.h>
#include <stdint.h>
#include <string.h>

typedef int NVENCSTATUS;
#define NV_ENC_SUCCESS 0

/* Simplified function list — we only check that the vtable is populated.
 * The real struct has version + 41 function pointers. */
typedef struct {
    uint32_t version;
    uint32_t reserved;
    void* funcs[41];
} NV_ENCODE_API_FUNCTION_LIST;

typedef NVENCSTATUS (*pfn_NvEncodeAPIGetMaxSupportedVersion)(uint32_t*);
typedef NVENCSTATUS (*pfn_NvEncodeAPICreateInstance)(NV_ENCODE_API_FUNCTION_LIST*);

static int pass = 0, fail = 0;

#define CHECK(name, expr) do { \
    NVENCSTATUS r = (expr); \
    if (r == NV_ENC_SUCCESS) { printf("  PASS: %s\n", name); pass++; } \
    else { printf("  FAIL: %s (error %d)\n", name, r); fail++; } \
} while(0)

int main(void) {
    printf("=== NVENC Interpose Test ===\n");

    void* lib = dlopen("librgpu_nvenc_interpose.so", RTLD_NOW);
    if (!lib) {
        printf("FAIL: dlopen: %s\n", dlerror());
        return 1;
    }

    pfn_NvEncodeAPIGetMaxSupportedVersion getVersion =
        (pfn_NvEncodeAPIGetMaxSupportedVersion)dlsym(lib, "NvEncodeAPIGetMaxSupportedVersion");
    pfn_NvEncodeAPICreateInstance createInstance =
        (pfn_NvEncodeAPICreateInstance)dlsym(lib, "NvEncodeAPICreateInstance");

    if (!getVersion || !createInstance) {
        printf("FAIL: missing symbols\n");
        dlclose(lib);
        return 1;
    }

    /* 1. Get max supported version */
    uint32_t version = 0;
    CHECK("NvEncodeAPIGetMaxSupportedVersion", getVersion(&version));
    printf("  INFO: max version = %u.%u\n", version >> 4, version & 0xF);

    /* 2. Create instance (get vtable) */
    NV_ENCODE_API_FUNCTION_LIST func_list;
    memset(&func_list, 0, sizeof(func_list));
    /* Set version field — NVENC expects NV_ENCODE_API_FUNCTION_LIST_VER */
    func_list.version = (uint32_t)(sizeof(NV_ENCODE_API_FUNCTION_LIST)) | (12 << 24);
    CHECK("NvEncodeAPICreateInstance", createInstance(&func_list));

    /* 3. Verify some vtable entries are populated (non-null) */
    int populated = 0;
    for (int i = 0; i < 41; i++) {
        if (func_list.funcs[i] != NULL) populated++;
    }
    if (populated > 0) {
        printf("  PASS: vtable has %d populated entries\n", populated);
        pass++;
    } else {
        printf("  FAIL: vtable is empty\n");
        fail++;
    }

    dlclose(lib);

    printf("=== NVENC: %d passed, %d failed ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
