/*
 * rgpu-vgpu.c - RGPU Virtual GPU Display Adapter Driver
 *
 * Minimal Windows KMDF root-enumerated driver that creates virtual GPU
 * device nodes under "Display adapters" in Device Manager. The driver
 * itself is a placeholder — actual GPU work is handled by user-mode
 * interpose libraries (CUDA, Vulkan, NVENC, NVDEC, NVML).
 *
 * Hardware ID: Root\RGPU_VGPU
 * Class: Display ({4d36e968-e325-11ce-bfc1-08002be10318})
 *
 * The Rust device_manager.rs creates/removes device instances via
 * SetupDi APIs. Each instance gets a friendly name like:
 *   "NVIDIA GeForce RTX 3070 (Remote - RGPU)"
 *
 * Copyright (c) 2026 RGPU Project
 * Licensed under MIT OR Apache-2.0
 */

#include <ntddk.h>
#include <wdf.h>
#include <initguid.h>
#include <devguid.h>
#include <ntstrsafe.h>

/* ------------------------------------------------------------------ */
/* Driver tag for memory allocations                                   */
/* ------------------------------------------------------------------ */
#define RGPU_POOL_TAG  'UPGR'

/* ------------------------------------------------------------------ */
/* Device context structure                                            */
/* ------------------------------------------------------------------ */
typedef struct _RGPU_DEVICE_CONTEXT {
    WDFDEVICE   Device;
    ULONG       InstanceIndex;
} RGPU_DEVICE_CONTEXT, *PRGPU_DEVICE_CONTEXT;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(RGPU_DEVICE_CONTEXT, RgpuGetDeviceContext)

/* ------------------------------------------------------------------ */
/* Forward declarations                                                */
/* ------------------------------------------------------------------ */
DRIVER_INITIALIZE               DriverEntry;
EVT_WDF_DRIVER_DEVICE_ADD       EvtDeviceAdd;
EVT_WDF_DEVICE_PREPARE_HARDWARE EvtDevicePrepareHardware;
EVT_WDF_DEVICE_RELEASE_HARDWARE EvtDeviceReleaseHardware;
EVT_WDF_DEVICE_D0_ENTRY         EvtDeviceD0Entry;
EVT_WDF_DEVICE_D0_EXIT          EvtDeviceD0Exit;

/* ------------------------------------------------------------------ */
/* EvtDevicePrepareHardware                                            */
/*                                                                     */
/* Called when the device is started. For a software-only virtual       */
/* device there is no real hardware to prepare, so this is a no-op.    */
/* ------------------------------------------------------------------ */
NTSTATUS
EvtDevicePrepareHardware(
    _In_ WDFDEVICE    Device,
    _In_ WDFCMRESLIST ResourcesRaw,
    _In_ WDFCMRESLIST ResourcesTranslated
    )
{
    UNREFERENCED_PARAMETER(Device);
    UNREFERENCED_PARAMETER(ResourcesRaw);
    UNREFERENCED_PARAMETER(ResourcesTranslated);

    KdPrintEx((DPFLTR_IHVDRIVER_ID, DPFLTR_INFO_LEVEL,
               "RGPU-VGPU: PrepareHardware\n"));

    return STATUS_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* EvtDeviceReleaseHardware                                            */
/*                                                                     */
/* Called when the device is stopped or removed. No-op for virtual hw. */
/* ------------------------------------------------------------------ */
NTSTATUS
EvtDeviceReleaseHardware(
    _In_ WDFDEVICE    Device,
    _In_ WDFCMRESLIST ResourcesTranslated
    )
{
    UNREFERENCED_PARAMETER(Device);
    UNREFERENCED_PARAMETER(ResourcesTranslated);

    KdPrintEx((DPFLTR_IHVDRIVER_ID, DPFLTR_INFO_LEVEL,
               "RGPU-VGPU: ReleaseHardware\n"));

    return STATUS_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* EvtDeviceD0Entry                                                    */
/*                                                                     */
/* Device is entering D0 (powered on) state.                           */
/* ------------------------------------------------------------------ */
NTSTATUS
EvtDeviceD0Entry(
    _In_ WDFDEVICE              Device,
    _In_ WDF_POWER_STATE        PreviousState
    )
{
    UNREFERENCED_PARAMETER(Device);
    UNREFERENCED_PARAMETER(PreviousState);

    KdPrintEx((DPFLTR_IHVDRIVER_ID, DPFLTR_INFO_LEVEL,
               "RGPU-VGPU: D0Entry (power on)\n"));

    return STATUS_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* EvtDeviceD0Exit                                                     */
/*                                                                     */
/* Device is leaving D0 (powering down) state.                         */
/* ------------------------------------------------------------------ */
NTSTATUS
EvtDeviceD0Exit(
    _In_ WDFDEVICE              Device,
    _In_ WDF_POWER_STATE        TargetState
    )
{
    UNREFERENCED_PARAMETER(Device);
    UNREFERENCED_PARAMETER(TargetState);

    KdPrintEx((DPFLTR_IHVDRIVER_ID, DPFLTR_INFO_LEVEL,
               "RGPU-VGPU: D0Exit (power off)\n"));

    return STATUS_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* ReadFriendlyNameFromRegistry                                        */
/*                                                                     */
/* Reads a "FriendlyName" value from the device's hardware registry    */
/* key. The Rust device_manager sets this before enabling the device.  */
/* If not found, returns a default name.                               */
/* ------------------------------------------------------------------ */
static NTSTATUS
ReadFriendlyNameFromRegistry(
    _In_  WDFDEVICE  Device,
    _Out_ WCHAR     *Buffer,
    _In_  ULONG      BufferLength
    )
{
    NTSTATUS    status;
    WDFKEY      hKey = NULL;
    UNICODE_STRING valueName;

    DECLARE_UNICODE_STRING_SIZE(friendlyName, 256);

    RtlInitUnicodeString(&valueName, L"FriendlyName");

    status = WdfDeviceOpenRegistryKey(
        Device,
        PLUGPLAY_REGKEY_DEVICE,
        KEY_READ,
        WDF_NO_OBJECT_ATTRIBUTES,
        &hKey
    );

    if (!NT_SUCCESS(status)) {
        goto UseDefault;
    }

    status = WdfRegistryQueryUnicodeString(hKey, &valueName, NULL, &friendlyName);
    WdfRegistryClose(hKey);

    if (!NT_SUCCESS(status) || friendlyName.Length == 0) {
        goto UseDefault;
    }

    /* Copy the string to output buffer */
    RtlStringCbCopyW(Buffer, BufferLength, friendlyName.Buffer);
    return STATUS_SUCCESS;

UseDefault:
    RtlStringCbCopyW(Buffer, BufferLength, L"RGPU Virtual GPU");
    return STATUS_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* EvtDeviceAdd                                                        */
/*                                                                     */
/* Called for each device instance enumerated under Root\RGPU_VGPU.    */
/* Creates the WDF device object with PnP power callbacks.             */
/* Sets the device friendly name from the registry.                    */
/* ------------------------------------------------------------------ */
NTSTATUS
EvtDeviceAdd(
    _In_    WDFDRIVER       Driver,
    _Inout_ PWDFDEVICE_INIT DeviceInit
    )
{
    NTSTATUS                        status;
    WDFDEVICE                       device;
    PRGPU_DEVICE_CONTEXT            deviceContext;
    WDF_OBJECT_ATTRIBUTES           deviceAttributes;
    WDF_PNPPOWER_EVENT_CALLBACKS    pnpPowerCallbacks;
    WCHAR                           friendlyName[256];

    UNREFERENCED_PARAMETER(Driver);

    KdPrintEx((DPFLTR_IHVDRIVER_ID, DPFLTR_INFO_LEVEL,
               "RGPU-VGPU: EvtDeviceAdd\n"));

    /* ----- PnP / Power callbacks ----- */
    WDF_PNPPOWER_EVENT_CALLBACKS_INIT(&pnpPowerCallbacks);
    pnpPowerCallbacks.EvtDevicePrepareHardware = EvtDevicePrepareHardware;
    pnpPowerCallbacks.EvtDeviceReleaseHardware = EvtDeviceReleaseHardware;
    pnpPowerCallbacks.EvtDeviceD0Entry         = EvtDeviceD0Entry;
    pnpPowerCallbacks.EvtDeviceD0Exit          = EvtDeviceD0Exit;

    WdfDeviceInitSetPnpPowerEventCallbacks(DeviceInit, &pnpPowerCallbacks);

    /* ----- Device context ----- */
    WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&deviceAttributes, RGPU_DEVICE_CONTEXT);

    /* ----- Create the device ----- */
    status = WdfDeviceCreate(&DeviceInit, &deviceAttributes, &device);
    if (!NT_SUCCESS(status)) {
        KdPrintEx((DPFLTR_IHVDRIVER_ID, DPFLTR_ERROR_LEVEL,
                   "RGPU-VGPU: WdfDeviceCreate failed 0x%x\n", status));
        return status;
    }

    /* ----- Initialize device context ----- */
    deviceContext = RgpuGetDeviceContext(device);
    deviceContext->Device = device;
    deviceContext->InstanceIndex = 0;

    /* ----- Set friendly name from registry ----- */
    status = ReadFriendlyNameFromRegistry(device, friendlyName, sizeof(friendlyName));
    if (NT_SUCCESS(status)) {
        UNICODE_STRING usFriendlyName;
        RtlInitUnicodeString(&usFriendlyName, friendlyName);

        /* Set the device description which shows in Device Manager */
        status = WdfDeviceAssignProperty(
            device,
            &DEVPKEY_Device_FriendlyName,
            DEVPROP_TYPE_STRING,
            (ULONG)(wcslen(friendlyName) + 1) * sizeof(WCHAR),
            friendlyName
        );

        if (!NT_SUCCESS(status)) {
            /* Non-fatal: device will use the INF description instead */
            KdPrintEx((DPFLTR_IHVDRIVER_ID, DPFLTR_WARNING_LEVEL,
                       "RGPU-VGPU: WdfDeviceAssignProperty(FriendlyName) "
                       "failed 0x%x\n", status));
        }
    }

    KdPrintEx((DPFLTR_IHVDRIVER_ID, DPFLTR_INFO_LEVEL,
               "RGPU-VGPU: Device created successfully: %ws\n",
               friendlyName));

    return STATUS_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* DriverEntry                                                         */
/*                                                                     */
/* Standard KMDF driver entry point. Creates the WDFDRIVER object      */
/* and registers EvtDeviceAdd.                                         */
/* ------------------------------------------------------------------ */
NTSTATUS
DriverEntry(
    _In_ PDRIVER_OBJECT  DriverObject,
    _In_ PUNICODE_STRING RegistryPath
    )
{
    NTSTATUS            status;
    WDF_DRIVER_CONFIG   config;

    KdPrintEx((DPFLTR_IHVDRIVER_ID, DPFLTR_INFO_LEVEL,
               "RGPU-VGPU: DriverEntry - RGPU Virtual GPU Driver v1.0\n"));

    WDF_DRIVER_CONFIG_INIT(&config, EvtDeviceAdd);

    status = WdfDriverCreate(
        DriverObject,
        RegistryPath,
        WDF_NO_OBJECT_ATTRIBUTES,
        &config,
        WDF_NO_HANDLE
    );

    if (!NT_SUCCESS(status)) {
        KdPrintEx((DPFLTR_IHVDRIVER_ID, DPFLTR_ERROR_LEVEL,
                   "RGPU-VGPU: WdfDriverCreate failed 0x%x\n", status));
        return status;
    }

    KdPrintEx((DPFLTR_IHVDRIVER_ID, DPFLTR_INFO_LEVEL,
               "RGPU-VGPU: Driver initialized successfully\n"));

    return STATUS_SUCCESS;
}
