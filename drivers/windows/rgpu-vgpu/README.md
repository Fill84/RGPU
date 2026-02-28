# RGPU Virtual GPU Display Adapter Driver

Minimal Windows KMDF kernel driver that creates virtual GPU device nodes
in Windows Device Manager under "Display adapters".

## What This Driver Does

This is a **root-enumerated software device** — it does not control any real
hardware. Its sole purpose is to create device nodes that appear in Device
Manager so that remote GPUs shared via RGPU look like locally installed GPUs.

The actual GPU compute/rendering work is handled entirely by user-mode
interpose libraries (CUDA, Vulkan, NVENC, NVDEC, NVML) that intercept API
calls and forward them over the network to the remote GPU server.

**Hardware ID:** `Root\RGPU_VGPU`
**Device Class:** Display (`{4d36e968-e325-11ce-bfc1-08002be10318}`)

## How It Works

1. The driver registers as a display adapter class device
2. The Rust `device_manager.rs` (in the RGPU client daemon) uses Windows
   SetupDi APIs to create/remove device instances dynamically
3. Each instance gets a friendly name set via the registry, e.g.:
   `"NVIDIA GeForce RTX 3070 (Remote - RGPU)"`
4. The PnP manager loads the driver for each instance, which then shows
   up under "Display adapters" in Device Manager

## Prerequisites

- **Visual Studio 2022** with "Desktop development with C++" workload
- **Windows SDK** (matching your WDK version)
- **Windows Driver Kit (WDK)** 10.0.26100.x or later
  - Download: https://learn.microsoft.com/en-us/windows-hardware/drivers/download-the-wdk
  - Make sure to install the WDK Visual Studio extension

## Building

From a Developer Command Prompt for VS 2022:

```cmd
cd drivers\windows\rgpu-vgpu
build.cmd release
```

Or open `rgpu-vgpu.vcxproj` in Visual Studio (generated on first build).

The build script will:
1. Locate the WDK and MSBuild automatically
2. Generate a `.vcxproj` if one doesn't exist
3. Build the driver, producing `build\Release\rgpu-vgpu.sys`

## Installation

### Step 1: Enable Test Signing (Development Only)

For development, you need to enable test signing since the driver is
not signed by Microsoft (WHQL) or an EV certificate:

```cmd
bcdedit /set testsigning on
```

Reboot after changing this setting.

### Step 2: Install the Driver Package

Using `pnputil` (run as Administrator):

```cmd
pnputil /add-driver rgpu-vgpu.inf /install
```

Or using `devcon` (from the WDK tools):

```cmd
devcon install rgpu-vgpu.inf Root\RGPU_VGPU
```

### Step 3: Create Device Instances

The RGPU client daemon creates device instances automatically via SetupDi
APIs when remote GPUs connect. You can also create them manually:

```cmd
devcon install rgpu-vgpu.inf Root\RGPU_VGPU
```

Each call creates a new device instance under "Display adapters".

## How the Rust device_manager.rs Manages Instances

The Rust code in `crates/rgpu-client/src/device_manager.rs` performs these
operations via the `windows-sys` crate and SetupDi APIs:

### Creating a device instance:

```text
1. SetupDiCreateDeviceInfoList(GUID_DEVCLASS_DISPLAY)
2. SetupDiCreateDeviceInfo("RGPU Virtual GPU")
3. SetupDiSetDeviceRegistryProperty(SPDRP_HARDWAREID, "Root\RGPU_VGPU")
4. SetupDiSetDeviceRegistryProperty(SPDRP_FRIENDLYNAME, "NVIDIA GeForce RTX 3070 (Remote - RGPU)")
5. SetupDiCallClassInstaller(DIF_REGISTERDEVICE)
6. Write "FriendlyName" to device hardware registry key
7. Trigger driver install via DeviceInstallation or devcon
```

### Removing a device instance:

```text
1. SetupDiGetClassDevs(GUID_DEVCLASS_DISPLAY)
2. Enumerate to find the target instance
3. SetupDiCallClassInstaller(DIF_REMOVE)
4. SetupDiDestroyDeviceInfoList()
```

## Signing

### Test Signing (Development)

1. Create a test certificate:
   ```cmd
   makecert -r -pe -ss PrivateCertStore -n "CN=RGPU Test" rgpu-test.cer
   ```

2. Sign the driver:
   ```cmd
   signtool sign /v /s PrivateCertStore /n "RGPU Test" /t http://timestamp.digicert.com rgpu-vgpu.sys
   ```

3. Create and sign the catalog:
   ```cmd
   inf2cat /driver:. /os:10_X64
   signtool sign /v /s PrivateCertStore /n "RGPU Test" /t http://timestamp.digicert.com rgpu-vgpu.cat
   ```

### Production Signing (WHQL / EV Certificate)

For production deployment:

1. **Attestation Signing** via the Windows Hardware Developer Center:
   - Create a Microsoft Partner Center account
   - Submit the driver package for attestation signing
   - This signs the driver with Microsoft's signature

2. **EV Code Signing Certificate**:
   - Purchase from a CA (DigiCert, Sectigo, etc.)
   - Required for Partner Center submission
   - Sign the driver package before submission

3. **WHQL Testing** (optional, for Windows certification):
   - Run the HLK (Hardware Lab Kit) tests
   - Submit results with the driver to Microsoft

## Uninstallation

Remove all device instances:
```cmd
devcon remove Root\RGPU_VGPU
```

Remove the driver package:
```cmd
pnputil /delete-driver rgpu-vgpu.inf /uninstall
```

## Troubleshooting

**Device shows with yellow exclamation mark:**
- Check that test signing is enabled: `bcdedit | findstr testsigning`
- Verify the .sys file is properly signed
- Check Event Viewer > System for driver load errors

**Device doesn't appear after devcon install:**
- Run as Administrator
- Check `devcon status Root\RGPU_VGPU` for status
- Look in Device Manager > View > Show hidden devices

**Build fails with "WDK not found":**
- Install WDK from the link in Prerequisites
- Make sure the WDK VS extension is installed
- Try building from a "Developer Command Prompt for VS 2022"

## File Structure

```
drivers/windows/rgpu-vgpu/
  rgpu-vgpu.c        Source code (KMDF driver)
  rgpu-vgpu.inf      Installation INF file
  rgpu-vgpu.cat      Catalog file (created by signing)
  rgpu-vgpu.vcxproj  VS project file (auto-generated)
  build.cmd           Build script
  README.md           This file
  build/
    Debug/
      rgpu-vgpu.sys   Debug driver binary
    Release/
      rgpu-vgpu.sys   Release driver binary
```

## License

Copyright (c) 2026 RGPU Project. Licensed under MIT OR Apache-2.0.
