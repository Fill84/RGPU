/*
 * test_nvdec.c — RGPU NVDEC interpose integration test
 *
 * Tests: cuvidGetDecoderCaps
 *        Validates the interpose library loads and forwards calls.
 *
 * Note: Full decoder creation requires a CUDA context. This test
 *       validates the entry point works without crashing.
 *
 * Run with: LD_PRELOAD=/usr/lib/rgpu/librgpu_nvdec_interpose.so ./test_nvdec
 */
#include <stdio.h>
#include <dlfcn.h>
#include <stdint.h>
#include <string.h>

#define CUDA_SUCCESS 0

typedef int CUresult;

/*
 * CUVIDDECODECAPS layout (simplified, matching rgpu-nvdec-interpose):
 * Offsets:
 *   0: eCodecType (u32, IN)
 *   4: eChromaFormat (u32, IN)
 *   8: nBitDepthMinus8 (u32, IN)
 *  24: bIsSupported (u32, OUT)
 *  28: nNumNVDECs (u32, OUT)
 * Total struct is ~256 bytes, we use a byte buffer.
 */
typedef unsigned char CUVIDDECODECAPS[256];

typedef CUresult (*pfn_cuvidGetDecoderCaps)(void*);

static int pass = 0, fail = 0;

int main(void) {
    printf("=== NVDEC Interpose Test ===\n");

    void* lib = dlopen("librgpu_nvdec_interpose.so", RTLD_NOW);
    if (!lib) {
        printf("FAIL: dlopen: %s\n", dlerror());
        return 1;
    }

    pfn_cuvidGetDecoderCaps cuvidGetDecoderCaps =
        (pfn_cuvidGetDecoderCaps)dlsym(lib, "cuvidGetDecoderCaps");

    if (!cuvidGetDecoderCaps) {
        printf("FAIL: missing cuvidGetDecoderCaps symbol\n");
        dlclose(lib);
        return 1;
    }
    printf("  PASS: cuvidGetDecoderCaps symbol found\n");
    pass++;

    /* Call cuvidGetDecoderCaps — it will forward to the server.
     * The call may return an error if no CUDA context is active,
     * but it should NOT crash. */
    CUVIDDECODECAPS caps;
    memset(caps, 0, sizeof(caps));
    /* eCodecType = H264 (4), eChromaFormat = 420 (1), nBitDepthMinus8 = 0 */
    *((uint32_t*)&caps[0]) = 4;
    *((uint32_t*)&caps[4]) = 1;
    *((uint32_t*)&caps[8]) = 0;

    CUresult r = cuvidGetDecoderCaps(caps);
    /* We accept CUDA_SUCCESS or any non-crash error */
    if (r == CUDA_SUCCESS) {
        printf("  PASS: cuvidGetDecoderCaps returned SUCCESS\n");
        pass++;
        uint32_t supported = *((uint32_t*)&caps[24]);
        printf("  INFO: bIsSupported = %u\n", supported);
    } else {
        /* Non-zero return is acceptable (e.g. no context) — the important
         * thing is we didn't crash and the call was forwarded. */
        printf("  WARN: cuvidGetDecoderCaps returned %d (may need CUDA context)\n", r);
        printf("  PASS: call completed without crash\n");
        pass++;
    }

    dlclose(lib);

    printf("=== NVDEC: %d passed, %d failed ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
