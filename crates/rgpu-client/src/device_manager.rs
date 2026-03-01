//! Virtual GPU device manager for OS-level GPU visibility.
//! Creates/removes virtual device nodes so remote GPUs appear in
//! Device Manager (Windows) or /dev/ (Linux).
//!
//! On Windows: uses SetupDi APIs to register devices under the Display
//! adapter class ({4d36e968-e325-11ce-bfc1-08002be10318}) so they appear
//! in Device Manager under "Display adapters".
//!
//! On Linux: uses ioctl on /dev/rgpu_control to communicate with the
//! kernel module.

use tracing::{info, warn, debug};

use rgpu_protocol::gpu_info::GpuInfo;
use crate::pool_manager::LOCAL_SERVER_ID;

/// Manages virtual GPU device nodes in the OS.
pub struct DeviceManager {
    /// Currently registered virtual GPU devices
    devices: Vec<VirtualGpuDevice>,
}

struct VirtualGpuDevice {
    name: String,
    index: u32,
    total_memory: u64,
    /// Windows: the PnP device instance ID (e.g. "ROOT\RGPU_VGPU\0000")
    /// Used for removal via SetupDi.
    #[cfg(windows)]
    device_instance_id: Option<String>,
}

impl DeviceManager {
    pub fn new() -> Self {
        let mut dm = Self {
            devices: Vec::new(),
        };
        dm.cleanup_orphaned_devices();
        dm
    }

    /// Sync virtual devices with the current GPU list.
    /// Adds new devices, removes disconnected ones.
    /// Only remote GPUs get virtual device nodes (local GPUs already have real ones).
    pub fn sync_devices(&mut self, gpus: &[GpuInfo]) {
        // Filter to only remote GPUs (server_id != LOCAL_SERVER_ID)
        let remote_gpus: Vec<&GpuInfo> = gpus
            .iter()
            .filter(|gpu| gpu.server_id != LOCAL_SERVER_ID)
            .collect();

        // Find GPUs that are no longer present and need removal
        let indices_to_remove: Vec<u32> = self
            .devices
            .iter()
            .filter(|dev| {
                !remote_gpus.iter().any(|gpu| {
                    gpu.server_device_index == dev.index
                        && gpu.device_name == dev.name
                })
            })
            .map(|dev| dev.index)
            .collect();

        for index in &indices_to_remove {
            self.remove_device(*index);
        }
        self.devices.retain(|dev| !indices_to_remove.contains(&dev.index));

        // Find GPUs that are new and need to be added
        for gpu in &remote_gpus {
            let already_exists = self.devices.iter().any(|dev| {
                dev.index == gpu.server_device_index && dev.name == gpu.device_name
            });

            if !already_exists {
                let friendly_name = format!("{} (Remote - RGPU)", gpu.device_name);
                let vdev = self.create_device(
                    &friendly_name,
                    gpu.server_device_index,
                    gpu.total_memory,
                );
                if let Some(vdev) = vdev {
                    info!(
                        "virtual device created: {} (index={}, memory={}MB)",
                        vdev.name,
                        vdev.index,
                        vdev.total_memory / (1024 * 1024)
                    );
                    self.devices.push(vdev);
                }
            }
        }
    }

    /// Remove all virtual device nodes (cleanup on shutdown).
    pub fn remove_all(&mut self) {
        let indices: Vec<u32> = self.devices.iter().map(|d| d.index).collect();
        for index in indices {
            self.remove_device(index);
        }
        self.devices.clear();
        info!("all virtual GPU device nodes removed");
    }

    /// Remove any orphaned RGPU virtual devices from previous runs.
    /// Called on startup to ensure a clean state.
    fn cleanup_orphaned_devices(&mut self) {
        #[cfg(windows)]
        platform::cleanup_orphaned();
        // Linux: kernel module handles cleanup on module unload
    }
}

// ============================================================================
// Windows implementation — SetupDi (Display adapter class)
// ============================================================================

#[cfg(windows)]
mod platform {
    use super::*;

    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        SetupDiCreateDeviceInfoList, SetupDiCreateDeviceInfoW,
        SetupDiSetDeviceRegistryPropertyW, SetupDiCallClassInstaller,
        SetupDiGetDeviceInstanceIdW, SetupDiDestroyDeviceInfoList,
        SetupDiGetClassDevsW, SetupDiEnumDeviceInfo,
        SetupDiGetDeviceRegistryPropertyW, SetupDiOpenDeviceInfoW,
        SetupDiRemoveDevice,
        SP_DEVINFO_DATA, DICD_GENERATE_ID,
        SPDRP_HARDWAREID, SPDRP_FRIENDLYNAME,
        DIF_REGISTERDEVICE,
    };
    use windows_sys::core::GUID;

    /// Display adapter class GUID: {4d36e968-e325-11ce-bfc1-08002be10318}
    const GUID_DEVCLASS_DISPLAY: GUID = GUID {
        data1: 0x4d36e968,
        data2: 0xe325,
        data3: 0x11ce,
        data4: [0xbf, 0xc1, 0x08, 0x00, 0x2b, 0xe1, 0x03, 0x18],
    };

    /// Hardware ID for RGPU virtual GPU devices.
    const RGPU_HARDWARE_ID: &str = "RGPU\\VirtualGPU";

    /// Encode a Rust string as a null-terminated wide string (UTF-16).
    fn to_wide_null(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// Encode a string as a double-null-terminated multi-sz string.
    fn to_multi_sz(s: &str) -> Vec<u16> {
        s.encode_utf16()
            .chain(std::iter::once(0))
            .chain(std::iter::once(0))
            .collect()
    }

    /// Check if a device's hardware IDs contain our RGPU hardware ID.
    unsafe fn device_has_rgpu_hwid(
        dev_info_set: isize,
        dev_info_data: &mut SP_DEVINFO_DATA,
    ) -> bool {
        let mut buf = [0u16; 512];
        let mut required_size = 0u32;
        let result = SetupDiGetDeviceRegistryPropertyW(
            dev_info_set,
            dev_info_data,
            SPDRP_HARDWAREID,
            std::ptr::null_mut(),
            buf.as_mut_ptr() as *mut u8,
            (buf.len() * 2) as u32,
            &mut required_size,
        );
        if result == 0 {
            return false;
        }

        // Multi-sz: scan null-separated strings for our hardware ID
        let target: Vec<u16> = RGPU_HARDWARE_ID.encode_utf16().collect();
        let mut start = 0;
        for (i, &ch) in buf.iter().enumerate() {
            if ch == 0 {
                if i > start {
                    let slice = &buf[start..i];
                    if slice == target.as_slice() {
                        return true;
                    }
                }
                start = i + 1;
                // Double null = end of multi-sz
                if start < buf.len() && buf[start] == 0 {
                    break;
                }
            }
        }
        false
    }

    /// Remove orphaned RGPU virtual GPU devices from previous runs.
    /// Scans all Display class devices and removes any with our hardware ID.
    pub fn cleanup_orphaned() {
        unsafe {
            let dev_info_set = SetupDiGetClassDevsW(
                &GUID_DEVCLASS_DISPLAY,
                std::ptr::null(),
                std::ptr::null_mut(),
                0, // All devices in class (including non-present/phantom)
            );
            if dev_info_set as isize == -1 {
                debug!("SetupDiGetClassDevsW failed during cleanup — no orphans to remove");
                return;
            }

            let mut removed_count = 0u32;
            let mut enum_index = 0u32;
            loop {
                let mut dev_info_data: SP_DEVINFO_DATA = std::mem::zeroed();
                dev_info_data.cbSize = std::mem::size_of::<SP_DEVINFO_DATA>() as u32;

                if SetupDiEnumDeviceInfo(dev_info_set, enum_index, &mut dev_info_data) == 0 {
                    break;
                }
                enum_index += 1;

                if device_has_rgpu_hwid(dev_info_set, &mut dev_info_data) {
                    if SetupDiRemoveDevice(dev_info_set, &mut dev_info_data) != 0 {
                        removed_count += 1;
                    }
                }
            }

            SetupDiDestroyDeviceInfoList(dev_info_set);

            if removed_count > 0 {
                info!("cleaned up {} orphaned RGPU virtual device(s) from previous run", removed_count);
            }
        }
    }

    impl DeviceManager {
        /// Create a virtual GPU device node under "Display adapters" via SetupDi.
        ///
        /// Uses DIF_REGISTERDEVICE to register the device with PnP under the
        /// Display class. The device appears in Device Manager under
        /// "Display adapters". No driver installation is attempted — the device
        /// will show status code 28 (no driver) which is expected for a
        /// virtual proxy device.
        pub(super) fn create_device(
            &self,
            friendly_name: &str,
            index: u32,
            total_memory: u64,
        ) -> Option<VirtualGpuDevice> {
            unsafe {
                // Step 1: Create device info set for Display class
                let dev_info_set = SetupDiCreateDeviceInfoList(
                    &GUID_DEVCLASS_DISPLAY,
                    std::ptr::null_mut(),
                );
                if dev_info_set as isize == -1 {
                    warn!(
                        "SetupDiCreateDeviceInfoList failed: {}; virtual device not created",
                        std::io::Error::last_os_error()
                    );
                    return None;
                }

                // Step 2: Create device info element
                // "RGPU_VGPU" with DICD_GENERATE_ID → Windows generates ROOT\RGPU_VGPU\xxxx
                let dev_name_wide = to_wide_null("RGPU_VGPU");
                let desc_wide = to_wide_null(friendly_name);
                let mut dev_info_data: SP_DEVINFO_DATA = std::mem::zeroed();
                dev_info_data.cbSize = std::mem::size_of::<SP_DEVINFO_DATA>() as u32;

                let result = SetupDiCreateDeviceInfoW(
                    dev_info_set,
                    dev_name_wide.as_ptr(),
                    &GUID_DEVCLASS_DISPLAY,
                    desc_wide.as_ptr(),
                    std::ptr::null_mut(),
                    DICD_GENERATE_ID,
                    &mut dev_info_data,
                );
                if result == 0 {
                    let err = std::io::Error::last_os_error();
                    warn!(
                        "SetupDiCreateDeviceInfoW failed: {}; virtual device not created \
                         -- the interpose libraries still work independently",
                        err
                    );
                    SetupDiDestroyDeviceInfoList(dev_info_set);
                    return None;
                }

                // Step 3: Set hardware ID (multi-sz string)
                let hw_ids = to_multi_sz(RGPU_HARDWARE_ID);
                let result = SetupDiSetDeviceRegistryPropertyW(
                    dev_info_set,
                    &mut dev_info_data,
                    SPDRP_HARDWAREID,
                    hw_ids.as_ptr() as *const u8,
                    (hw_ids.len() * 2) as u32,
                );
                if result == 0 {
                    let err = std::io::Error::last_os_error();
                    warn!(
                        "SetupDiSetDeviceRegistryPropertyW(HARDWAREID) failed: {}",
                        err
                    );
                    SetupDiDestroyDeviceInfoList(dev_info_set);
                    return None;
                }

                // Step 4: Set friendly name (shown in Device Manager)
                let friendly_wide = to_wide_null(friendly_name);
                let _ = SetupDiSetDeviceRegistryPropertyW(
                    dev_info_set,
                    &mut dev_info_data,
                    SPDRP_FRIENDLYNAME,
                    friendly_wide.as_ptr() as *const u8,
                    (friendly_wide.len() * 2) as u32,
                );

                // Step 5: Register device with PnP (DIF_REGISTERDEVICE)
                // This creates the device node under Display adapters.
                // We intentionally do NOT call DIF_INSTALLDEVICE — that requires
                // a signed driver. The device appears with code 28 (no driver)
                // but IS visible under Display adapters.
                let result = SetupDiCallClassInstaller(
                    DIF_REGISTERDEVICE,
                    dev_info_set,
                    &mut dev_info_data,
                );
                if result == 0 {
                    let err = std::io::Error::last_os_error();
                    warn!(
                        "SetupDiCallClassInstaller(DIF_REGISTERDEVICE) failed: {}; \
                         virtual device not created -- the interpose libraries still work independently",
                        err
                    );
                    SetupDiDestroyDeviceInfoList(dev_info_set);
                    return None;
                }

                // Step 6: Read back the generated device instance ID
                let mut instance_id_buf = [0u16; 256];
                let mut required_size = 0u32;
                SetupDiGetDeviceInstanceIdW(
                    dev_info_set,
                    &mut dev_info_data,
                    instance_id_buf.as_mut_ptr(),
                    instance_id_buf.len() as u32,
                    &mut required_size,
                );
                let instance_id_str = String::from_utf16_lossy(
                    &instance_id_buf[..required_size.saturating_sub(1) as usize]
                );

                info!(
                    "virtual GPU registered under Display adapters: {} (instance: {})",
                    friendly_name, instance_id_str
                );

                SetupDiDestroyDeviceInfoList(dev_info_set);

                Some(VirtualGpuDevice {
                    name: friendly_name.to_string(),
                    index,
                    total_memory,
                    device_instance_id: Some(instance_id_str),
                })
            }
        }

        /// Remove a virtual GPU device node via SetupDi.
        pub(super) fn remove_device(&self, index: u32) {
            let device = match self.devices.iter().find(|d| d.index == index) {
                Some(d) => d,
                None => return,
            };

            let instance_id = match &device.device_instance_id {
                Some(id) => id,
                None => return,
            };

            unsafe {
                let dev_info_set = SetupDiCreateDeviceInfoList(
                    &GUID_DEVCLASS_DISPLAY,
                    std::ptr::null_mut(),
                );
                if dev_info_set as isize == -1 {
                    warn!("SetupDiCreateDeviceInfoList failed during removal");
                    return;
                }

                let instance_id_wide = to_wide_null(instance_id);
                let mut dev_info_data: SP_DEVINFO_DATA = std::mem::zeroed();
                dev_info_data.cbSize = std::mem::size_of::<SP_DEVINFO_DATA>() as u32;

                let result = SetupDiOpenDeviceInfoW(
                    dev_info_set,
                    instance_id_wide.as_ptr(),
                    std::ptr::null_mut(),
                    0,
                    &mut dev_info_data,
                );
                if result == 0 {
                    let err = std::io::Error::last_os_error();
                    warn!(
                        "SetupDiOpenDeviceInfoW failed for {}: {}",
                        instance_id, err
                    );
                    SetupDiDestroyDeviceInfoList(dev_info_set);
                    return;
                }

                let result = SetupDiRemoveDevice(dev_info_set, &mut dev_info_data);
                if result == 0 {
                    let err = std::io::Error::last_os_error();
                    warn!(
                        "SetupDiRemoveDevice failed for {}: {}",
                        instance_id, err
                    );
                } else {
                    info!(
                        "removed virtual GPU from Display adapters: {} (index={})",
                        device.name, index
                    );
                }

                SetupDiDestroyDeviceInfoList(dev_info_set);
            }
        }
    }
}

// ============================================================================
// Linux implementation
// ============================================================================

#[cfg(unix)]
mod platform {
    use super::*;
    use std::fs::OpenOptions;
    use std::os::unix::io::AsRawFd;

    const RGPU_IOCTL_MAGIC: u8 = b'R';
    const RGPU_CONTROL_PATH: &str = "/dev/rgpu_control";

    /// Struct layout matching the kernel module's `rgpu_gpu_info`.
    /// Must be kept in sync with the kernel module definition.
    #[repr(C)]
    struct RgpuGpuInfo {
        /// GPU name, null-terminated, max 128 bytes
        name: [u8; 128],
        /// Total VRAM in bytes
        total_memory: u64,
        /// Device index
        index: u32,
        /// Padding for alignment
        _pad: u32,
    }

    /// _IOW(type, nr, size) for Linux ioctl number generation.
    /// Direction: _IOW = 1 (write from userspace to kernel).
    /// Format: direction(2) | size(14) | type(8) | nr(8)
    const fn ioctl_iow(magic: u8, nr: u8, size: usize) -> u64 {
        let dir: u64 = 1; // _IOC_WRITE
        ((dir << 30) | ((size as u64 & 0x3FFF) << 16) | ((magic as u64) << 8) | (nr as u64))
    }

    const RGPU_IOCTL_ADD_GPU: u64 =
        ioctl_iow(RGPU_IOCTL_MAGIC, 1, std::mem::size_of::<RgpuGpuInfo>());
    const RGPU_IOCTL_REMOVE_GPU: u64 =
        ioctl_iow(RGPU_IOCTL_MAGIC, 2, std::mem::size_of::<u32>());

    impl DeviceManager {
        /// Create a virtual GPU device node via ioctl to /dev/rgpu_control.
        pub(super) fn create_device(
            &self,
            friendly_name: &str,
            index: u32,
            total_memory: u64,
        ) -> Option<VirtualGpuDevice> {
            let file = match OpenOptions::new().write(true).open(RGPU_CONTROL_PATH) {
                Ok(f) => f,
                Err(e) => {
                    warn!(
                        "failed to open {}: {}; \
                         virtual device not created -- the interpose libraries still work independently",
                        RGPU_CONTROL_PATH, e
                    );
                    return None;
                }
            };

            let mut gpu_info = RgpuGpuInfo {
                name: [0u8; 128],
                total_memory,
                index,
                _pad: 0,
            };

            // Copy the name, truncating to 127 bytes (leaving room for null terminator)
            let name_bytes = friendly_name.as_bytes();
            let copy_len = name_bytes.len().min(127);
            gpu_info.name[..copy_len].copy_from_slice(&name_bytes[..copy_len]);

            let fd = file.as_raw_fd();
            let ret = unsafe {
                libc::ioctl(
                    fd,
                    RGPU_IOCTL_ADD_GPU as libc::c_ulong,
                    &gpu_info as *const RgpuGpuInfo,
                )
            };

            if ret < 0 {
                warn!(
                    "ioctl RGPU_IOCTL_ADD_GPU failed (error {}): {}; \
                     virtual device not created",
                    std::io::Error::last_os_error().raw_os_error().unwrap_or(-1),
                    std::io::Error::last_os_error()
                );
                return None;
            }

            Some(VirtualGpuDevice {
                name: friendly_name.to_string(),
                index,
                total_memory,
            })
        }

        /// Remove a virtual GPU device node via ioctl to /dev/rgpu_control.
        pub(super) fn remove_device(&self, index: u32) {
            let file = match OpenOptions::new().write(true).open(RGPU_CONTROL_PATH) {
                Ok(f) => f,
                Err(e) => {
                    warn!(
                        "failed to open {} for device removal: {}",
                        RGPU_CONTROL_PATH, e
                    );
                    return;
                }
            };

            let fd = file.as_raw_fd();
            let ret = unsafe {
                libc::ioctl(
                    fd,
                    RGPU_IOCTL_REMOVE_GPU as libc::c_ulong,
                    &index as *const u32,
                )
            };

            if ret < 0 {
                warn!(
                    "ioctl RGPU_IOCTL_REMOVE_GPU failed for index {} (error {}): {}",
                    index,
                    std::io::Error::last_os_error().raw_os_error().unwrap_or(-1),
                    std::io::Error::last_os_error()
                );
            } else {
                info!("removed virtual GPU device node for index {}", index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_device_manager() {
        let dm = DeviceManager::new();
        assert!(dm.devices.is_empty());
    }

    #[test]
    fn test_sync_filters_local_gpus() {
        let mut dm = DeviceManager::new();

        // Create a GPU with LOCAL_SERVER_ID -- should be filtered out
        let local_gpu = GpuInfo {
            device_name: "Local GPU".to_string(),
            vendor_id: 0x10de,
            device_id: 0x1234,
            device_type: rgpu_protocol::gpu_info::GpuDeviceType::DiscreteGpu,
            total_memory: 8 * 1024 * 1024 * 1024,
            supports_vulkan: true,
            supports_cuda: true,
            vulkan_api_version: None,
            vulkan_driver_version: None,
            cuda_compute_capability: None,
            queue_family_count: 1,
            memory_heaps: vec![],
            server_device_index: 0,
            server_id: LOCAL_SERVER_ID,
        };

        // Sync with only local GPUs -- should create no virtual devices
        dm.sync_devices(&[local_gpu]);

        // No devices should have been added because local GPUs are filtered out
        assert!(dm.devices.is_empty());
    }

    #[test]
    fn test_remove_all_clears_devices() {
        let mut dm = DeviceManager::new();
        dm.remove_all();
        assert!(dm.devices.is_empty());
    }
}
