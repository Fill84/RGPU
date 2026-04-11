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
