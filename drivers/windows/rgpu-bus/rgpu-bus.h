/*
 * rgpu-bus.h - RGPU Virtual Bus Driver shared definitions
 *
 * Defines IOCTL codes and structures shared between the kernel driver
 * and the user-mode RGPU daemon.
 */
#ifndef RGPU_BUS_H
#define RGPU_BUS_H

#include <winioctl.h>

#define FILE_DEVICE_RGPU_BUS  0x8000

#define IOCTL_RGPU_ADD_GPU \
    CTL_CODE(FILE_DEVICE_RGPU_BUS, 0x800, METHOD_BUFFERED, FILE_WRITE_ACCESS)

#define IOCTL_RGPU_REMOVE_GPU \
    CTL_CODE(FILE_DEVICE_RGPU_BUS, 0x801, METHOD_BUFFERED, FILE_WRITE_ACCESS)

#define IOCTL_RGPU_LIST_GPUS \
    CTL_CODE(FILE_DEVICE_RGPU_BUS, 0x802, METHOD_BUFFERED, FILE_READ_ACCESS)

#define RGPU_MAX_GPUS 16

typedef struct _RGPU_GPU_INFO {
    ULONG  VendorId;
    ULONG  DeviceId;
    ULONG  SubsystemId;
    UCHAR  Revision;
    WCHAR  DeviceName[128];
    ULONG64 TotalMemory;
    ULONG  SlotIndex;
} RGPU_GPU_INFO, *PRGPU_GPU_INFO;

typedef struct _RGPU_GPU_REMOVE {
    ULONG SlotIndex;
} RGPU_GPU_REMOVE, *PRGPU_GPU_REMOVE;

typedef struct _RGPU_GPU_LIST {
    ULONG Count;
    RGPU_GPU_INFO Gpus[RGPU_MAX_GPUS];
} RGPU_GPU_LIST, *PRGPU_GPU_LIST;

#endif /* RGPU_BUS_H */
