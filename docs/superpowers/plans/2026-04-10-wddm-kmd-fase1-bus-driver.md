# WDDM KMD Fase 1A: Virtual Bus Driver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Windows kernel-mode virtual bus driver (`rgpu-bus.sys`) that creates and removes child PCI-like devices on demand, enabling hot-swap of remote GPUs via IOCTLs from the RGPU daemon.

**Architecture:** WDM bus driver that enumerates child PDOs with NVIDIA hardware IDs (PCI\VEN_10DE&DEV_XXXX). The daemon communicates via IOCTLs on a control device (`\\.\RGPUBus`). When a remote GPU connects, the daemon tells the bus to add a child device; Windows PnP detects the new device and loads the appropriate function driver. On disconnect, the child is removed.

**Tech Stack:** C (WDM kernel driver), WDK 10.0.26100.0, Visual Studio 2022, INF, test-signing

**Spec:** `docs/superpowers/specs/2026-04-10-wddm-kmd-remote-gpu-design.md`

**Important:** This is Plan A of 3. Plan B (WDDM miniport) and Plan C (daemon extensions) follow. This plan delivers a testable bus driver that can add/remove child devices visible in Device Manager.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `drivers/windows/rgpu-bus/rgpu-bus.c` | Create | Bus driver: DriverEntry, AddDevice, PnP, IOCTL handling, child PDO management |
| `drivers/windows/rgpu-bus/rgpu-bus.h` | Create | Shared types: IOCTL codes, GPU info struct |
| `drivers/windows/rgpu-bus/rgpu-bus.inf` | Create | INF installer for root-enumerated bus driver |
| `drivers/windows/rgpu-bus/Makefile` | Create | Build script using WDK MSBuild or NMake |
| `drivers/windows/rgpu-bus/rgpu-bus.vcxproj` | Create | Visual Studio project for WDK build |
| `drivers/windows/rgpu-bus/install.cmd` | Create | Test-signing + install script |

---

### Task 1: Project Setup — WDK Build Infrastructure

**Files:**
- Create: `drivers/windows/rgpu-bus/rgpu-bus.vcxproj`
- Create: `drivers/windows/rgpu-bus/rgpu-bus.h`
- Create: `drivers/windows/rgpu-bus/Makefile`

- [ ] **Step 1: Create the project directory**

```bash
mkdir -p drivers/windows/rgpu-bus
```

- [ ] **Step 2: Create rgpu-bus.h with shared types and IOCTL definitions**

```c
/*
 * rgpu-bus.h - RGPU Virtual Bus Driver shared definitions
 *
 * Defines IOCTL codes and structures shared between the kernel driver
 * and the user-mode RGPU daemon.
 */
#ifndef RGPU_BUS_H
#define RGPU_BUS_H

#include <winioctl.h>

/* Device type for RGPU bus IOCTLs */
#define FILE_DEVICE_RGPU_BUS  0x8000

/* IOCTL codes */
#define IOCTL_RGPU_ADD_GPU \
    CTL_CODE(FILE_DEVICE_RGPU_BUS, 0x800, METHOD_BUFFERED, FILE_WRITE_ACCESS)

#define IOCTL_RGPU_REMOVE_GPU \
    CTL_CODE(FILE_DEVICE_RGPU_BUS, 0x801, METHOD_BUFFERED, FILE_WRITE_ACCESS)

#define IOCTL_RGPU_LIST_GPUS \
    CTL_CODE(FILE_DEVICE_RGPU_BUS, 0x802, METHOD_BUFFERED, FILE_READ_ACCESS)

/* Maximum number of remote GPUs the bus can enumerate */
#define RGPU_MAX_GPUS 16

/* GPU info passed from daemon to bus driver via IOCTL_RGPU_ADD_GPU */
typedef struct _RGPU_GPU_INFO {
    ULONG  VendorId;         /* PCI vendor ID (0x10DE for NVIDIA) */
    ULONG  DeviceId;         /* PCI device ID (e.g. 0x2484 for RTX 3070) */
    ULONG  SubsystemId;      /* PCI subsystem ID */
    UCHAR  Revision;         /* PCI revision */
    WCHAR  DeviceName[128];  /* Human-readable GPU name */
    ULONG64 TotalMemory;     /* Total VRAM in bytes */
    ULONG  SlotIndex;        /* Assigned slot (0-15), returned by driver */
} RGPU_GPU_INFO, *PRGPU_GPU_INFO;

/* GPU removal request — just the slot index */
typedef struct _RGPU_GPU_REMOVE {
    ULONG SlotIndex;
} RGPU_GPU_REMOVE, *PRGPU_GPU_REMOVE;

/* GPU list response */
typedef struct _RGPU_GPU_LIST {
    ULONG Count;
    RGPU_GPU_INFO Gpus[RGPU_MAX_GPUS];
} RGPU_GPU_LIST, *PRGPU_GPU_LIST;

#endif /* RGPU_BUS_H */
```

- [ ] **Step 3: Create the Visual Studio WDK project file**

```xml
<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" ToolsVersion="12.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Debug|x64">
      <Configuration>Debug</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release|x64">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label="Globals">
    <ProjectGuid>{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}</ProjectGuid>
    <RootNamespace>rgpu_bus</RootNamespace>
    <DriverType>WDM</DriverType>
    <TargetVersion>Windows10</TargetVersion>
    <ProjectName>rgpu-bus</ProjectName>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  <PropertyGroup Label="Configuration" Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <TargetVersion>Windows10</TargetVersion>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>WindowsKernelModeDriver10.0</PlatformToolset>
    <ConfigurationType>Driver</ConfigurationType>
    <DriverTargetPlatform>Universal</DriverTargetPlatform>
  </PropertyGroup>
  <PropertyGroup Label="Configuration" Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <TargetVersion>Windows10</TargetVersion>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>WindowsKernelModeDriver10.0</PlatformToolset>
    <ConfigurationType>Driver</ConfigurationType>
    <DriverTargetPlatform>Universal</DriverTargetPlatform>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" />
  <ItemGroup>
    <ClCompile Include="rgpu-bus.c" />
    <ClInclude Include="rgpu-bus.h" />
  </ItemGroup>
  <ItemGroup>
    <Inf Include="rgpu-bus.inf" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />
</Project>
```

- [ ] **Step 4: Create Makefile for command-line builds**

```makefile
# Makefile for rgpu-bus.sys
# Requires: Visual Studio 2022 + WDK 10.0.26100.0
#
# Usage:
#   nmake           (from a "x64 Native Tools Command Prompt for VS 2022")
#   nmake clean
#   nmake install   (requires admin + test-signing enabled)

SOLUTION_DIR = .
PROJECT = rgpu-bus

all:
	msbuild $(PROJECT).vcxproj /p:Configuration=Debug /p:Platform=x64

release:
	msbuild $(PROJECT).vcxproj /p:Configuration=Release /p:Platform=x64

clean:
	msbuild $(PROJECT).vcxproj /t:Clean /p:Configuration=Debug /p:Platform=x64
	-rmdir /s /q x64

install: all
	@echo === Installing rgpu-bus driver (test-signed) ===
	@echo Ensure test-signing is enabled: bcdedit -set TESTSIGNING ON
	devcon install rgpu-bus.inf Root\RGPUBus
	@echo === Done ===

uninstall:
	devcon remove Root\RGPUBus
	@echo === Uninstalled ===
```

- [ ] **Step 5: Commit**

```bash
git add drivers/windows/rgpu-bus/
git commit -m "feat: WDK project setup for rgpu-bus virtual bus driver"
```

---

### Task 2: Bus Driver Core — DriverEntry and FDO Setup

**Files:**
- Create: `drivers/windows/rgpu-bus/rgpu-bus.c`

This is the main driver source file. We build it incrementally — this task creates the skeleton with DriverEntry, AddDevice, and PnP dispatch.

- [ ] **Step 1: Write the bus driver skeleton**

```c
/*
 * rgpu-bus.c — RGPU Virtual Bus Driver
 *
 * A WDM bus driver that creates child PDOs (Physical Device Objects)
 * representing remote GPUs. The RGPU daemon communicates via IOCTLs
 * to add/remove child devices dynamically (hot-swap).
 *
 * Child devices report NVIDIA hardware IDs so that Windows loads
 * the appropriate display driver (rgpu-kmd.sys) for them.
 *
 * Copyright (c) 2026 RGPU Project
 * Licensed under MIT OR Apache-2.0
 */

#include <ntddk.h>
#include <wdmsec.h>
#include <ntstrsafe.h>

/* Include shared IOCTL definitions (kernel-safe subset) */
/* We redefine the structures here for kernel use */

#define RGPU_MAX_GPUS 16
#define RGPU_POOL_TAG 'bGPR'

/* ------------------------------------------------------------------ */
/* IOCTL definitions                                                   */
/* ------------------------------------------------------------------ */

#define FILE_DEVICE_RGPU_BUS  0x8000

#define IOCTL_RGPU_ADD_GPU \
    CTL_CODE(FILE_DEVICE_RGPU_BUS, 0x800, METHOD_BUFFERED, FILE_WRITE_ACCESS)

#define IOCTL_RGPU_REMOVE_GPU \
    CTL_CODE(FILE_DEVICE_RGPU_BUS, 0x801, METHOD_BUFFERED, FILE_WRITE_ACCESS)

#define IOCTL_RGPU_LIST_GPUS \
    CTL_CODE(FILE_DEVICE_RGPU_BUS, 0x802, METHOD_BUFFERED, FILE_READ_ACCESS)

/* ------------------------------------------------------------------ */
/* Structures                                                          */
/* ------------------------------------------------------------------ */

typedef struct _RGPU_GPU_INFO {
    ULONG     VendorId;
    ULONG     DeviceId;
    ULONG     SubsystemId;
    UCHAR     Revision;
    WCHAR     DeviceName[128];
    ULONG64   TotalMemory;
    ULONG     SlotIndex;
} RGPU_GPU_INFO, *PRGPU_GPU_INFO;

typedef struct _RGPU_GPU_REMOVE {
    ULONG SlotIndex;
} RGPU_GPU_REMOVE, *PRGPU_GPU_REMOVE;

typedef struct _RGPU_GPU_LIST {
    ULONG Count;
    RGPU_GPU_INFO Gpus[RGPU_MAX_GPUS];
} RGPU_GPU_LIST, *PRGPU_GPU_LIST;

/* Per-child-device (PDO) context */
typedef struct _RGPU_PDO_CONTEXT {
    BOOLEAN        Active;
    ULONG          SlotIndex;
    PDEVICE_OBJECT Pdo;
    RGPU_GPU_INFO  GpuInfo;
} RGPU_PDO_CONTEXT, *PRGPU_PDO_CONTEXT;

/* Bus FDO (Functional Device Object) extension */
typedef struct _RGPU_BUS_EXTENSION {
    PDEVICE_OBJECT      Self;          /* This FDO */
    PDEVICE_OBJECT      LowerDevice;   /* PDO from PnP */
    PDEVICE_OBJECT      PhysicalDevice;
    FAST_MUTEX          Mutex;         /* Protects Slots[] */
    RGPU_PDO_CONTEXT    Slots[RGPU_MAX_GPUS];
    ULONG               GpuCount;
} RGPU_BUS_EXTENSION, *PRGPU_BUS_EXTENSION;

/* ------------------------------------------------------------------ */
/* Forward declarations                                                */
/* ------------------------------------------------------------------ */

DRIVER_INITIALIZE            DriverEntry;
DRIVER_ADD_DEVICE            BusAddDevice;
DRIVER_UNLOAD                BusUnload;

__drv_dispatchType(IRP_MJ_PNP)
DRIVER_DISPATCH              BusPnp;

__drv_dispatchType(IRP_MJ_DEVICE_CONTROL)
DRIVER_DISPATCH              BusIoctl;

__drv_dispatchType(IRP_MJ_CREATE)
__drv_dispatchType(IRP_MJ_CLOSE)
DRIVER_DISPATCH              BusCreateClose;

static NTSTATUS BusHandleQueryDeviceRelations(
    _In_ PRGPU_BUS_EXTENSION BusExt,
    _Inout_ PIRP Irp);

static NTSTATUS BusAddGpu(
    _In_ PRGPU_BUS_EXTENSION BusExt,
    _In_ PRGPU_GPU_INFO GpuInfo);

static NTSTATUS BusRemoveGpu(
    _In_ PRGPU_BUS_EXTENSION BusExt,
    _In_ ULONG SlotIndex);

/* ------------------------------------------------------------------ */
/* DriverEntry                                                         */
/* ------------------------------------------------------------------ */

NTSTATUS
DriverEntry(
    _In_ PDRIVER_OBJECT  DriverObject,
    _In_ PUNICODE_STRING RegistryPath
    )
{
    UNREFERENCED_PARAMETER(RegistryPath);

    DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL,
               "RGPU Bus: DriverEntry\n");

    DriverObject->DriverExtension->AddDevice     = BusAddDevice;
    DriverObject->DriverUnload                    = BusUnload;
    DriverObject->MajorFunction[IRP_MJ_PNP]      = BusPnp;
    DriverObject->MajorFunction[IRP_MJ_DEVICE_CONTROL] = BusIoctl;
    DriverObject->MajorFunction[IRP_MJ_CREATE]    = BusCreateClose;
    DriverObject->MajorFunction[IRP_MJ_CLOSE]     = BusCreateClose;

    return STATUS_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* BusAddDevice                                                        */
/* ------------------------------------------------------------------ */

NTSTATUS
BusAddDevice(
    _In_ PDRIVER_OBJECT DriverObject,
    _In_ PDEVICE_OBJECT PhysicalDeviceObject
    )
{
    NTSTATUS            status;
    PDEVICE_OBJECT      fdo = NULL;
    PRGPU_BUS_EXTENSION busExt;
    UNICODE_STRING      devName;
    UNICODE_STRING      symLink;

    DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL,
               "RGPU Bus: AddDevice\n");

    RtlInitUnicodeString(&devName, L"\\Device\\RGPUBus");
    RtlInitUnicodeString(&symLink, L"\\DosDevices\\RGPUBus");

    status = IoCreateDevice(
        DriverObject,
        sizeof(RGPU_BUS_EXTENSION),
        &devName,
        FILE_DEVICE_BUS_EXTENDER,
        FILE_DEVICE_SECURE_OPEN,
        FALSE,
        &fdo);

    if (!NT_SUCCESS(status)) {
        DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_ERROR_LEVEL,
                   "RGPU Bus: IoCreateDevice failed 0x%08X\n", status);
        return status;
    }

    status = IoCreateSymbolicLink(&symLink, &devName);
    if (!NT_SUCCESS(status)) {
        IoDeleteDevice(fdo);
        return status;
    }

    busExt = (PRGPU_BUS_EXTENSION)fdo->DeviceExtension;
    RtlZeroMemory(busExt, sizeof(RGPU_BUS_EXTENSION));
    busExt->Self = fdo;
    busExt->PhysicalDevice = PhysicalDeviceObject;
    ExInitializeFastMutex(&busExt->Mutex);

    busExt->LowerDevice = IoAttachDeviceToDeviceStack(fdo, PhysicalDeviceObject);
    if (busExt->LowerDevice == NULL) {
        IoDeleteSymbolicLink(&symLink);
        IoDeleteDevice(fdo);
        return STATUS_NO_SUCH_DEVICE;
    }

    fdo->Flags &= ~DO_DEVICE_INITIALIZING;
    fdo->Flags |= DO_BUFFERED_IO;

    DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL,
               "RGPU Bus: AddDevice complete\n");

    return STATUS_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* BusUnload                                                           */
/* ------------------------------------------------------------------ */

VOID
BusUnload(
    _In_ PDRIVER_OBJECT DriverObject
    )
{
    UNREFERENCED_PARAMETER(DriverObject);
    DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL,
               "RGPU Bus: Unload\n");
}

/* ------------------------------------------------------------------ */
/* BusCreateClose — Allow opening/closing the control device           */
/* ------------------------------------------------------------------ */

NTSTATUS
BusCreateClose(
    _In_ PDEVICE_OBJECT DeviceObject,
    _Inout_ PIRP Irp
    )
{
    UNREFERENCED_PARAMETER(DeviceObject);
    Irp->IoStatus.Status = STATUS_SUCCESS;
    Irp->IoStatus.Information = 0;
    IoCompleteRequest(Irp, IO_NO_INCREMENT);
    return STATUS_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* BusPnp — PnP IRP dispatch                                          */
/* ------------------------------------------------------------------ */

NTSTATUS
BusPnp(
    _In_ PDEVICE_OBJECT DeviceObject,
    _Inout_ PIRP Irp
    )
{
    PRGPU_BUS_EXTENSION busExt;
    PIO_STACK_LOCATION  irpStack;
    NTSTATUS            status;

    busExt = (PRGPU_BUS_EXTENSION)DeviceObject->DeviceExtension;
    irpStack = IoGetCurrentIrpStackLocation(Irp);

    switch (irpStack->MinorFunction) {

    case IRP_MN_QUERY_DEVICE_RELATIONS:
        if (irpStack->Parameters.QueryDeviceRelations.Type == BusRelations) {
            status = BusHandleQueryDeviceRelations(busExt, Irp);
            return status;
        }
        break;

    case IRP_MN_START_DEVICE:
        Irp->IoStatus.Status = STATUS_SUCCESS;
        break;

    case IRP_MN_STOP_DEVICE:
    case IRP_MN_REMOVE_DEVICE:
        Irp->IoStatus.Status = STATUS_SUCCESS;
        if (irpStack->MinorFunction == IRP_MN_REMOVE_DEVICE) {
            UNICODE_STRING symLink;
            RtlInitUnicodeString(&symLink, L"\\DosDevices\\RGPUBus");
            IoDeleteSymbolicLink(&symLink);
            IoDetachDevice(busExt->LowerDevice);
            IoDeleteDevice(DeviceObject);
        }
        break;

    case IRP_MN_QUERY_CAPABILITIES: {
        /* Pass down first, then modify */
        IoSkipCurrentIrpStackLocation(Irp);
        status = IoCallDriver(busExt->LowerDevice, Irp);
        return status;
    }

    default:
        break;
    }

    IoSkipCurrentIrpStackLocation(Irp);
    return IoCallDriver(busExt->LowerDevice, Irp);
}

/* ------------------------------------------------------------------ */
/* BusHandleQueryDeviceRelations — Enumerate child PDOs                */
/* ------------------------------------------------------------------ */

static NTSTATUS
BusHandleQueryDeviceRelations(
    _In_ PRGPU_BUS_EXTENSION BusExt,
    _Inout_ PIRP Irp
    )
{
    PDEVICE_RELATIONS relations;
    ULONG             count = 0;
    ULONG             size;
    ULONG             i;

    ExAcquireFastMutex(&BusExt->Mutex);

    /* Count active children */
    for (i = 0; i < RGPU_MAX_GPUS; i++) {
        if (BusExt->Slots[i].Active && BusExt->Slots[i].Pdo != NULL)
            count++;
    }

    size = sizeof(DEVICE_RELATIONS) + (count > 0 ? (count - 1) : 0) * sizeof(PDEVICE_OBJECT);
    relations = (PDEVICE_RELATIONS)ExAllocatePool2(POOL_FLAG_PAGED, size, RGPU_POOL_TAG);

    if (relations == NULL) {
        ExReleaseFastMutex(&BusExt->Mutex);
        Irp->IoStatus.Status = STATUS_INSUFFICIENT_RESOURCES;
        IoCompleteRequest(Irp, IO_NO_INCREMENT);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    relations->Count = count;
    count = 0;
    for (i = 0; i < RGPU_MAX_GPUS; i++) {
        if (BusExt->Slots[i].Active && BusExt->Slots[i].Pdo != NULL) {
            ObReferenceObject(BusExt->Slots[i].Pdo);
            relations->Objects[count++] = BusExt->Slots[i].Pdo;
        }
    }

    ExReleaseFastMutex(&BusExt->Mutex);

    Irp->IoStatus.Information = (ULONG_PTR)relations;
    Irp->IoStatus.Status = STATUS_SUCCESS;
    IoSkipCurrentIrpStackLocation(Irp);
    return IoCallDriver(BusExt->LowerDevice, Irp);
}

/* ------------------------------------------------------------------ */
/* BusIoctl — Handle IOCTLs from daemon                                */
/* ------------------------------------------------------------------ */

NTSTATUS
BusIoctl(
    _In_ PDEVICE_OBJECT DeviceObject,
    _Inout_ PIRP Irp
    )
{
    PRGPU_BUS_EXTENSION busExt;
    PIO_STACK_LOCATION  irpStack;
    NTSTATUS            status = STATUS_INVALID_DEVICE_REQUEST;
    ULONG               inLen, outLen;
    PVOID               buffer;

    busExt = (PRGPU_BUS_EXTENSION)DeviceObject->DeviceExtension;
    irpStack = IoGetCurrentIrpStackLocation(Irp);
    inLen  = irpStack->Parameters.DeviceIoControl.InputBufferLength;
    outLen = irpStack->Parameters.DeviceIoControl.OutputBufferLength;
    buffer = Irp->AssociatedIrp.SystemBuffer;

    switch (irpStack->Parameters.DeviceIoControl.IoControlCode) {

    case IOCTL_RGPU_ADD_GPU:
        if (inLen >= sizeof(RGPU_GPU_INFO) && buffer != NULL) {
            PRGPU_GPU_INFO info = (PRGPU_GPU_INFO)buffer;
            status = BusAddGpu(busExt, info);
            if (NT_SUCCESS(status) && outLen >= sizeof(RGPU_GPU_INFO)) {
                /* Return the updated info (with assigned SlotIndex) */
                Irp->IoStatus.Information = sizeof(RGPU_GPU_INFO);
            }
        } else {
            status = STATUS_BUFFER_TOO_SMALL;
        }
        break;

    case IOCTL_RGPU_REMOVE_GPU:
        if (inLen >= sizeof(RGPU_GPU_REMOVE) && buffer != NULL) {
            PRGPU_GPU_REMOVE rem = (PRGPU_GPU_REMOVE)buffer;
            status = BusRemoveGpu(busExt, rem->SlotIndex);
        } else {
            status = STATUS_BUFFER_TOO_SMALL;
        }
        break;

    case IOCTL_RGPU_LIST_GPUS:
        if (outLen >= sizeof(RGPU_GPU_LIST) && buffer != NULL) {
            PRGPU_GPU_LIST list = (PRGPU_GPU_LIST)buffer;
            ULONG i;
            ExAcquireFastMutex(&busExt->Mutex);
            list->Count = 0;
            for (i = 0; i < RGPU_MAX_GPUS; i++) {
                if (busExt->Slots[i].Active) {
                    list->Gpus[list->Count] = busExt->Slots[i].GpuInfo;
                    list->Count++;
                }
            }
            ExReleaseFastMutex(&busExt->Mutex);
            Irp->IoStatus.Information = sizeof(RGPU_GPU_LIST);
            status = STATUS_SUCCESS;
        } else {
            status = STATUS_BUFFER_TOO_SMALL;
        }
        break;
    }

    Irp->IoStatus.Status = status;
    if (!NT_SUCCESS(status))
        Irp->IoStatus.Information = 0;
    IoCompleteRequest(Irp, IO_NO_INCREMENT);
    return status;
}

/* ------------------------------------------------------------------ */
/* Child PDO PnP dispatch                                              */
/* ------------------------------------------------------------------ */

static NTSTATUS
PdoPnp(
    _In_ PDEVICE_OBJECT DeviceObject,
    _Inout_ PIRP Irp
    )
{
    PRGPU_PDO_CONTEXT   pdoCtx;
    PIO_STACK_LOCATION  irpStack;
    NTSTATUS            status = STATUS_SUCCESS;

    pdoCtx = (PRGPU_PDO_CONTEXT)DeviceObject->DeviceExtension;
    irpStack = IoGetCurrentIrpStackLocation(Irp);

    switch (irpStack->MinorFunction) {

    case IRP_MN_QUERY_ID: {
        PWCHAR idBuf;
        ULONG  idLen;
        UNICODE_STRING idStr;

        switch (irpStack->Parameters.QueryId.IdType) {

        case BusQueryHardwareIDs: {
            /* Format: PCI\VEN_10DE&DEV_XXXX\0\0 (multi-sz) */
            WCHAR hwId[64];
            status = RtlStringCchPrintfW(hwId, 64,
                L"PCI\\VEN_%04X&DEV_%04X",
                pdoCtx->GpuInfo.VendorId,
                pdoCtx->GpuInfo.DeviceId);
            if (!NT_SUCCESS(status)) break;

            idLen = (ULONG)(wcslen(hwId) + 2) * sizeof(WCHAR);
            idBuf = (PWCHAR)ExAllocatePool2(POOL_FLAG_PAGED, idLen, RGPU_POOL_TAG);
            if (idBuf == NULL) { status = STATUS_INSUFFICIENT_RESOURCES; break; }
            RtlZeroMemory(idBuf, idLen);
            RtlCopyMemory(idBuf, hwId, wcslen(hwId) * sizeof(WCHAR));
            Irp->IoStatus.Information = (ULONG_PTR)idBuf;
            status = STATUS_SUCCESS;
            break;
        }

        case BusQueryCompatibleIDs: {
            /* Format: PCI\VEN_10DE\0\0 (multi-sz) */
            WCHAR compatId[32];
            status = RtlStringCchPrintfW(compatId, 32,
                L"PCI\\VEN_%04X", pdoCtx->GpuInfo.VendorId);
            if (!NT_SUCCESS(status)) break;

            idLen = (ULONG)(wcslen(compatId) + 2) * sizeof(WCHAR);
            idBuf = (PWCHAR)ExAllocatePool2(POOL_FLAG_PAGED, idLen, RGPU_POOL_TAG);
            if (idBuf == NULL) { status = STATUS_INSUFFICIENT_RESOURCES; break; }
            RtlZeroMemory(idBuf, idLen);
            RtlCopyMemory(idBuf, compatId, wcslen(compatId) * sizeof(WCHAR));
            Irp->IoStatus.Information = (ULONG_PTR)idBuf;
            status = STATUS_SUCCESS;
            break;
        }

        case BusQueryInstanceID: {
            /* Unique per-slot instance: "RGPU_SLOT_XX" */
            WCHAR instId[32];
            status = RtlStringCchPrintfW(instId, 32,
                L"RGPU_SLOT_%02u", pdoCtx->SlotIndex);
            if (!NT_SUCCESS(status)) break;

            idLen = (ULONG)(wcslen(instId) + 1) * sizeof(WCHAR);
            idBuf = (PWCHAR)ExAllocatePool2(POOL_FLAG_PAGED, idLen, RGPU_POOL_TAG);
            if (idBuf == NULL) { status = STATUS_INSUFFICIENT_RESOURCES; break; }
            RtlZeroMemory(idBuf, idLen);
            RtlCopyMemory(idBuf, instId, wcslen(instId) * sizeof(WCHAR));
            Irp->IoStatus.Information = (ULONG_PTR)idBuf;
            status = STATUS_SUCCESS;
            break;
        }

        case BusQueryDeviceSerialNumber:
            status = STATUS_NOT_SUPPORTED;
            break;

        default:
            status = STATUS_NOT_SUPPORTED;
            break;
        }
        break;
    }

    case IRP_MN_QUERY_DEVICE_TEXT: {
        if (irpStack->Parameters.QueryDeviceText.DeviceTextType == DeviceTextDescription) {
            ULONG len = (ULONG)(wcslen(pdoCtx->GpuInfo.DeviceName) + 1) * sizeof(WCHAR);
            PWCHAR desc = (PWCHAR)ExAllocatePool2(POOL_FLAG_PAGED, len, RGPU_POOL_TAG);
            if (desc) {
                RtlCopyMemory(desc, pdoCtx->GpuInfo.DeviceName, len);
                Irp->IoStatus.Information = (ULONG_PTR)desc;
                status = STATUS_SUCCESS;
            } else {
                status = STATUS_INSUFFICIENT_RESOURCES;
            }
        } else {
            status = STATUS_NOT_SUPPORTED;
        }
        break;
    }

    case IRP_MN_QUERY_CAPABILITIES: {
        PDEVICE_CAPABILITIES caps = irpStack->Parameters.DeviceCapabilities.Capabilities;
        caps->SilentInstall  = TRUE;
        caps->SurpriseRemovalOK = TRUE;
        caps->RawDeviceOK    = FALSE;
        caps->Removable      = TRUE;
        caps->EjectSupported = FALSE;
        caps->Address        = pdoCtx->SlotIndex;
        caps->UINumber       = pdoCtx->SlotIndex;
        status = STATUS_SUCCESS;
        break;
    }

    case IRP_MN_START_DEVICE:
        status = STATUS_SUCCESS;
        break;

    case IRP_MN_STOP_DEVICE:
    case IRP_MN_REMOVE_DEVICE:
        if (irpStack->MinorFunction == IRP_MN_REMOVE_DEVICE) {
            pdoCtx->Active = FALSE;
            /* Don't delete the PDO here — PnP manager owns it */
        }
        status = STATUS_SUCCESS;
        break;

    case IRP_MN_QUERY_REMOVE_DEVICE:
    case IRP_MN_CANCEL_REMOVE_DEVICE:
    case IRP_MN_SURPRISE_REMOVAL:
        status = STATUS_SUCCESS;
        break;

    case IRP_MN_QUERY_RESOURCE_REQUIREMENTS:
        /* No hardware resources needed */
        Irp->IoStatus.Information = 0;
        status = STATUS_SUCCESS;
        break;

    case IRP_MN_QUERY_BUS_INFORMATION: {
        PPNP_BUS_INFORMATION busInfo;
        busInfo = (PPNP_BUS_INFORMATION)ExAllocatePool2(
            POOL_FLAG_PAGED, sizeof(PNP_BUS_INFORMATION), RGPU_POOL_TAG);
        if (busInfo) {
            busInfo->BusTypeGuid = GUID_BUS_TYPE_INTERNAL;
            busInfo->LegacyBusType = PNPBus;
            busInfo->BusNumber = 0;
            Irp->IoStatus.Information = (ULONG_PTR)busInfo;
            status = STATUS_SUCCESS;
        } else {
            status = STATUS_INSUFFICIENT_RESOURCES;
        }
        break;
    }

    default:
        status = Irp->IoStatus.Status;
        break;
    }

    Irp->IoStatus.Status = status;
    IoCompleteRequest(Irp, IO_NO_INCREMENT);
    return status;
}

/* ------------------------------------------------------------------ */
/* BusAddGpu — Create a child PDO for a remote GPU                     */
/* ------------------------------------------------------------------ */

static NTSTATUS
BusAddGpu(
    _In_ PRGPU_BUS_EXTENSION BusExt,
    _In_ PRGPU_GPU_INFO GpuInfo
    )
{
    NTSTATUS        status;
    PDEVICE_OBJECT  pdo = NULL;
    PRGPU_PDO_CONTEXT pdoCtx;
    ULONG           slot;

    ExAcquireFastMutex(&BusExt->Mutex);

    /* Find free slot */
    slot = (ULONG)-1;
    for (ULONG i = 0; i < RGPU_MAX_GPUS; i++) {
        if (!BusExt->Slots[i].Active) {
            slot = i;
            break;
        }
    }

    if (slot == (ULONG)-1) {
        ExReleaseFastMutex(&BusExt->Mutex);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    /* Create child PDO */
    status = IoCreateDevice(
        BusExt->Self->DriverObject,
        sizeof(RGPU_PDO_CONTEXT),
        NULL,                           /* No device name for PDO */
        FILE_DEVICE_BUS_EXTENDER,
        FILE_AUTOGENERATED_DEVICE_NAME,
        FALSE,
        &pdo);

    if (!NT_SUCCESS(status)) {
        ExReleaseFastMutex(&BusExt->Mutex);
        return status;
    }

    /* Set up PDO dispatch — child PDOs handle their own PnP IRPs */
    pdo->DriverObject->MajorFunction[IRP_MJ_PNP] = PdoPnp;

    pdoCtx = (PRGPU_PDO_CONTEXT)pdo->DeviceExtension;
    pdoCtx->Active    = TRUE;
    pdoCtx->SlotIndex = slot;
    pdoCtx->Pdo       = pdo;
    pdoCtx->GpuInfo   = *GpuInfo;

    BusExt->Slots[slot] = *pdoCtx;
    BusExt->GpuCount++;

    /* Return assigned slot to caller */
    GpuInfo->SlotIndex = slot;

    pdo->Flags &= ~DO_DEVICE_INITIALIZING;

    ExReleaseFastMutex(&BusExt->Mutex);

    /* Tell PnP manager to re-enumerate our bus children */
    IoInvalidateDeviceRelations(BusExt->PhysicalDevice, BusRelations);

    DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL,
               "RGPU Bus: Added GPU slot %u: VEN_%04X DEV_%04X '%ws'\n",
               slot, GpuInfo->VendorId, GpuInfo->DeviceId, GpuInfo->DeviceName);

    return STATUS_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* BusRemoveGpu — Remove a child PDO                                   */
/* ------------------------------------------------------------------ */

static NTSTATUS
BusRemoveGpu(
    _In_ PRGPU_BUS_EXTENSION BusExt,
    _In_ ULONG SlotIndex
    )
{
    if (SlotIndex >= RGPU_MAX_GPUS)
        return STATUS_INVALID_PARAMETER;

    ExAcquireFastMutex(&BusExt->Mutex);

    if (!BusExt->Slots[SlotIndex].Active) {
        ExReleaseFastMutex(&BusExt->Mutex);
        return STATUS_NOT_FOUND;
    }

    /* Mark as inactive — PnP will query bus relations and see it's gone */
    BusExt->Slots[SlotIndex].Active = FALSE;

    if (BusExt->Slots[SlotIndex].Pdo != NULL) {
        IoDeleteDevice(BusExt->Slots[SlotIndex].Pdo);
        BusExt->Slots[SlotIndex].Pdo = NULL;
    }

    BusExt->GpuCount--;

    ExReleaseFastMutex(&BusExt->Mutex);

    /* Tell PnP manager to re-enumerate */
    IoInvalidateDeviceRelations(BusExt->PhysicalDevice, BusRelations);

    DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL,
               "RGPU Bus: Removed GPU slot %u\n", SlotIndex);

    return STATUS_SUCCESS;
}
```

- [ ] **Step 2: Verify it compiles**

Open "x64 Native Tools Command Prompt for VS 2022" and run:
```
cd drivers\windows\rgpu-bus
msbuild rgpu-bus.vcxproj /p:Configuration=Debug /p:Platform=x64
```

Expected: `rgpu-bus.sys` in `x64\Debug\`

- [ ] **Step 3: Commit**

```bash
git add drivers/windows/rgpu-bus/rgpu-bus.c
git commit -m "feat: rgpu-bus virtual bus driver — FDO, child PDO, IOCTL hot-swap"
```

---

### Task 3: INF Installer

**Files:**
- Create: `drivers/windows/rgpu-bus/rgpu-bus.inf`
- Create: `drivers/windows/rgpu-bus/install.cmd`

- [ ] **Step 1: Create the INF file**

```inf
;
; rgpu-bus.inf — RGPU Virtual Bus Driver
;
; Installs a root-enumerated bus driver that creates child devices
; representing remote GPUs.
;

[Version]
Signature   = "$WINDOWS NT$"
Class       = System
ClassGuid   = {4D36E97D-E325-11CE-BFC1-08002BE10318}
Provider    = %RGPU%
DriverVer   = 04/10/2026,1.0.0.0
CatalogFile = rgpu-bus.cat
PnpLockdown = 1

[DestinationDirs]
DefaultDestDir = 13

[SourceDisksNames]
1 = %DiskName%

[SourceDisksFiles]
rgpu-bus.sys = 1

[Manufacturer]
%RGPU% = RGPU,NTamd64

[RGPU.NTamd64]
%RGPUBus.DeviceDesc% = RGPUBus_Install, Root\RGPUBus

[RGPUBus_Install.NT]
CopyFiles = RGPUBus_CopyFiles

[RGPUBus_CopyFiles]
rgpu-bus.sys

[RGPUBus_Install.NT.Services]
AddService = RGPUBus, 0x00000002, RGPUBus_Service

[RGPUBus_Service]
DisplayName    = %RGPUBus.SvcDesc%
ServiceType    = 1               ; SERVICE_KERNEL_DRIVER
StartType      = 3               ; SERVICE_DEMAND_START
ErrorControl   = 1               ; SERVICE_ERROR_NORMAL
ServiceBinary  = %13%\rgpu-bus.sys

[Strings]
RGPU                = "RGPU Project"
RGPUBus.DeviceDesc  = "RGPU Virtual GPU Bus"
RGPUBus.SvcDesc     = "RGPU Virtual GPU Bus Driver"
DiskName            = "RGPU Driver Disk"
```

- [ ] **Step 2: Create install script**

```cmd
@echo off
REM install.cmd — Install rgpu-bus driver with test signing
REM Run as Administrator!

echo ============================================
echo  RGPU Bus Driver Installer (test-signed)
echo ============================================
echo.

REM Check admin
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Run this as Administrator!
    pause
    exit /b 1
)

REM Enable test signing if not already
bcdedit /set testsigning on >nul 2>&1

REM Self-sign the driver
set DRIVER_DIR=%~dp0x64\Debug
if not exist "%DRIVER_DIR%\rgpu-bus.sys" (
    echo ERROR: Build the driver first (msbuild)
    pause
    exit /b 1
)

REM Create test certificate and sign
makecert -r -pe -ss PrivateCertStore -n "CN=RGPU Test" RGPU_Test.cer >nul 2>&1
signtool sign /s PrivateCertStore /n "RGPU Test" /t http://timestamp.digicert.com /fd SHA256 "%DRIVER_DIR%\rgpu-bus.sys"

REM Copy INF to build dir
copy /y rgpu-bus.inf "%DRIVER_DIR%\" >nul

REM Install
echo Installing driver...
devcon install "%DRIVER_DIR%\rgpu-bus.inf" Root\RGPUBus

echo.
echo Done! Reboot if test-signing was just enabled.
pause
```

- [ ] **Step 3: Commit**

```bash
git add drivers/windows/rgpu-bus/rgpu-bus.inf drivers/windows/rgpu-bus/install.cmd
git commit -m "feat: rgpu-bus INF installer and test-signing install script"
```

---

### Task 4: Build, Sign, Install, and Test

This task is manual — it verifies the driver works on a test machine.

- [ ] **Step 1: Build the driver**

```
cd drivers\windows\rgpu-bus
msbuild rgpu-bus.vcxproj /p:Configuration=Debug /p:Platform=x64
```

- [ ] **Step 2: Enable test-signing (one-time, requires reboot)**

```
bcdedit -set TESTSIGNING ON
```
Reboot the machine.

- [ ] **Step 3: Install the driver**

Run `install.cmd` as Administrator.

Expected: devcon reports success, Device Manager shows "RGPU Virtual GPU Bus" under System Devices.

- [ ] **Step 4: Test IOCTL — Add a GPU**

Write a small test program (or use the RGPU daemon later) that opens `\\.\RGPUBus` and sends `IOCTL_RGPU_ADD_GPU` with:
- VendorId = 0x10DE
- DeviceId = 0x2484 (RTX 3070)
- DeviceName = L"NVIDIA GeForce RTX 3070"
- TotalMemory = 8589934592 (8GB)

Expected: Device Manager shows a new device under a category (might show as "Unknown device" with hardware ID `PCI\VEN_10DE&DEV_2484` until we install rgpu-kmd.sys).

- [ ] **Step 5: Test IOCTL — Remove GPU**

Send `IOCTL_RGPU_REMOVE_GPU` with the assigned SlotIndex.

Expected: Device disappears from Device Manager.

- [ ] **Step 6: Test IOCTL — List GPUs**

Send `IOCTL_RGPU_LIST_GPUS`.

Expected: Returns empty list (after removal) or populated list (after add).

- [ ] **Step 7: Commit any fixes**

```bash
git add -A
git commit -m "fix: bus driver fixes from first install test"
```

---

### Task 5: IOCTL Test Tool

**Files:**
- Create: `drivers/windows/rgpu-bus/test-bus.c`

A simple command-line tool to test the bus driver IOCTLs without needing the full RGPU daemon.

- [ ] **Step 1: Write test-bus.c**

```c
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
        printf("ERROR: Cannot open \\\\.\\ RGPUBus (error %lu)\n", GetLastError());
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
```

- [ ] **Step 2: Compile**

```
cl /W4 test-bus.c /Fe:test-bus.exe
```

- [ ] **Step 3: Test the full hot-swap cycle**

```cmd
test-bus list
REM Expected: GPUs: 0

test-bus add 10DE 2484 "NVIDIA GeForce RTX 3070" 8
REM Expected: SUCCESS: Added GPU at slot 0
REM Check Device Manager — new device should appear

test-bus list
REM Expected: GPUs: 1, shows RTX 3070

test-bus add 10DE 2204 "NVIDIA GeForce RTX 3090" 24
REM Expected: SUCCESS: Added GPU at slot 1

test-bus list
REM Expected: GPUs: 2

test-bus remove 0
REM Expected: SUCCESS: Removed GPU at slot 0
REM Check Device Manager — first device should disappear

test-bus list
REM Expected: GPUs: 1, shows RTX 3090 only
```

- [ ] **Step 4: Commit**

```bash
git add drivers/windows/rgpu-bus/test-bus.c
git commit -m "test: bus driver IOCTL test tool for hot-swap validation"
```
