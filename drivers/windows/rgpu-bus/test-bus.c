/*
 * test-bus.c — RGPU Bus Driver IOCTL test tool
 *
 * Usage:
 *   test-bus add <vendor_id> <device_id> <name> <memory_gb>
 *   test-bus remove <slot>
 *   test-bus list
 */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

/* Include shared header for IOCTL codes */
#include "rgpu-bus.h"

int wmain(int argc, wchar_t* argv[])
{
    HANDLE hDevice;

    hDevice = CreateFileW(
        L"\\\\.\\RGPUBus",
        GENERIC_READ | GENERIC_WRITE,
        0, NULL, OPEN_EXISTING, 0, NULL);

    if (hDevice == INVALID_HANDLE_VALUE) {
        printf("ERROR: Cannot open \\\\.\\RGPUBus (error %lu)\n", GetLastError());
        printf("Is the driver installed? Run install.cmd as admin.\n");
        return 1;
    }

    if (argc < 2) {
        printf("Usage:\n");
        printf("  test-bus add <vendor_id_hex> <device_id_hex> <name> <memory_gb>\n");
        printf("  test-bus remove <slot>\n");
        printf("  test-bus list\n");
        printf("\nExample:\n");
        printf("  test-bus add 10DE 2484 \"NVIDIA GeForce RTX 3070\" 8\n");
        CloseHandle(hDevice);
        return 1;
    }

    if (_wcsicmp(argv[1], L"add") == 0 && argc >= 6) {
        RGPU_GPU_INFO info = {0};
        DWORD bytesReturned;

        info.VendorId = (ULONG)wcstoul(argv[2], NULL, 16);
        info.DeviceId = (ULONG)wcstoul(argv[3], NULL, 16);
        wcsncpy_s(info.DeviceName, 128, argv[4], _TRUNCATE);
        info.TotalMemory = (ULONG64)_wtoi64(argv[5]) * 1024ULL * 1024ULL * 1024ULL;

        BOOL ok = DeviceIoControl(
            hDevice, IOCTL_RGPU_ADD_GPU,
            &info, sizeof(info),
            &info, sizeof(info),
            &bytesReturned, NULL);

        if (ok) {
            printf("SUCCESS: Added GPU at slot %u\n", info.SlotIndex);
            printf("  VEN_%04X DEV_%04X\n", info.VendorId, info.DeviceId);
            printf("  %ws\n", info.DeviceName);
            printf("  %llu MB VRAM\n", info.TotalMemory / (1024*1024));
        } else {
            printf("FAILED: error %lu\n", GetLastError());
        }
    }
    else if (_wcsicmp(argv[1], L"remove") == 0 && argc >= 3) {
        RGPU_GPU_REMOVE rem = {0};
        DWORD bytesReturned;

        rem.SlotIndex = (ULONG)_wtoi(argv[2]);

        BOOL ok = DeviceIoControl(
            hDevice, IOCTL_RGPU_REMOVE_GPU,
            &rem, sizeof(rem),
            NULL, 0,
            &bytesReturned, NULL);

        if (ok) {
            printf("SUCCESS: Removed GPU at slot %u\n", rem.SlotIndex);
        } else {
            printf("FAILED: error %lu\n", GetLastError());
        }
    }
    else if (_wcsicmp(argv[1], L"list") == 0) {
        RGPU_GPU_LIST list = {0};
        DWORD bytesReturned;

        BOOL ok = DeviceIoControl(
            hDevice, IOCTL_RGPU_LIST_GPUS,
            NULL, 0,
            &list, sizeof(list),
            &bytesReturned, NULL);

        if (ok) {
            printf("GPUs: %u\n", list.Count);
            for (ULONG i = 0; i < list.Count; i++) {
                printf("  [%u] VEN_%04X DEV_%04X %ws (%llu MB)\n",
                    list.Gpus[i].SlotIndex,
                    list.Gpus[i].VendorId,
                    list.Gpus[i].DeviceId,
                    list.Gpus[i].DeviceName,
                    list.Gpus[i].TotalMemory / (1024*1024));
            }
        } else {
            printf("FAILED: error %lu\n", GetLastError());
        }
    }
    else {
        printf("Unknown command: %ws\n", argv[1]);
    }

    CloseHandle(hDevice);
    return 0;
}
