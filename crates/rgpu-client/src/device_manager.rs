//! Virtual GPU device manager for OS-level GPU visibility.
//! Creates/removes virtual device nodes so remote GPUs appear in
//! Device Manager (Windows) or /dev/ (Linux).

use tracing::{info, warn};

use rgpu_protocol::gpu_info::GpuInfo;
use crate::pool_manager::LOCAL_SERVER_ID;

/// Manages virtual GPU device nodes in the OS.
/// On Windows: uses SetupDi APIs to create/remove device instances.
/// On Linux: uses ioctl on /dev/rgpu_control to communicate with the kernel module.
pub struct DeviceManager {
    /// Currently registered virtual GPU devices
    devices: Vec<VirtualGpuDevice>,
}

struct VirtualGpuDevice {
    name: String,
    index: u32,
    total_memory: u64,
    #[cfg(windows)]
    device_instance_id: Option<String>,
}

impl DeviceManager {
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
        }
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
}

// ============================================================================
// Windows implementation
// ============================================================================

#[cfg(windows)]
mod platform {
    use super::*;

    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        SetupDiCreateDeviceInfoList,
        SetupDiCreateDeviceInfoW,
        SetupDiSetDeviceRegistryPropertyW,
        SetupDiCallClassInstaller,
        SetupDiDestroyDeviceInfoList,
        SetupDiGetClassDevsW,
        SetupDiEnumDeviceInfo,
        SetupDiGetDeviceRegistryPropertyW,
        SetupDiRemoveDevice,
        CM_Get_Device_ID_Size_Ex,
        CM_Get_Device_IDW,
        SP_DEVINFO_DATA,
        DIGCF_PRESENT,
        DIGCF_PROFILE,
        DIF_REGISTERDEVICE,
        DIF_REMOVE,
        SPDRP_HARDWAREID,
        SPDRP_FRIENDLYNAME,
    };

    /// HDEVINFO invalid value (INVALID_HANDLE_VALUE = -1)
    const INVALID_HDEVINFO: isize = -1;

    /// GUID_DEVCLASS_DISPLAY = {4d36e968-e325-11ce-bfc1-08002be10318}
    const GUID_DEVCLASS_DISPLAY: windows_sys::core::GUID = windows_sys::core::GUID {
        data1: 0x4d36e968,
        data2: 0xe325,
        data3: 0x11ce,
        data4: [0xbf, 0xc1, 0x08, 0x00, 0x2b, 0xe1, 0x03, 0x18],
    };

    /// Hardware ID for our virtual GPU device. Must match the INF file.
    const HARDWARE_ID: &str = "Root\\RGPU_VGPU";

    /// DICD_GENERATE_ID — let SetupDi generate a unique device instance ID.
    const DICD_GENERATE_ID: u32 = 0x00000001;

    /// Encode a Rust string as a null-terminated wide string (UTF-16).
    fn to_wide_null(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// Encode a hardware ID as a double-null-terminated multi-sz string.
    fn to_multi_sz(s: &str) -> Vec<u16> {
        // Multi-sz format: string\0\0
        s.encode_utf16()
            .chain(std::iter::once(0))
            .chain(std::iter::once(0))
            .collect()
    }

    impl DeviceManager {
        /// Create a virtual GPU device node via SetupDi APIs.
        pub(super) fn create_device(
            &self,
            friendly_name: &str,
            index: u32,
            total_memory: u64,
        ) -> Option<VirtualGpuDevice> {
            unsafe {
                // Create an empty device info set for the Display class
                let dev_info_set = SetupDiCreateDeviceInfoList(
                    &GUID_DEVCLASS_DISPLAY as *const _,
                    std::ptr::null_mut(), // hwndParent
                );
                if dev_info_set == INVALID_HDEVINFO {
                    warn!(
                        "SetupDiCreateDeviceInfoList failed (error {}); \
                         virtual device not created -- the interpose libraries still work independently",
                        std::io::Error::last_os_error()
                    );
                    return None;
                }

                let mut dev_info_data = SP_DEVINFO_DATA {
                    cbSize: std::mem::size_of::<SP_DEVINFO_DATA>() as u32,
                    ClassGuid: GUID_DEVCLASS_DISPLAY,
                    DevInst: 0,
                    Reserved: 0,
                };

                // Create a device information element with hardware ID "Root\RGPU_VGPU"
                let device_name_wide = to_wide_null(HARDWARE_ID);
                let result = SetupDiCreateDeviceInfoW(
                    dev_info_set,
                    device_name_wide.as_ptr(),
                    &GUID_DEVCLASS_DISPLAY as *const _,
                    std::ptr::null(), // device description (set separately via friendly name)
                    std::ptr::null_mut(), // hwndParent
                    DICD_GENERATE_ID,
                    &mut dev_info_data,
                );
                if result == 0 {
                    warn!(
                        "SetupDiCreateDeviceInfoW failed (error {}); \
                         virtual device not created",
                        std::io::Error::last_os_error()
                    );
                    SetupDiDestroyDeviceInfoList(dev_info_set);
                    return None;
                }

                // Set the hardware ID (SPDRP_HARDWAREID) -- multi-sz format
                let hw_id_multi_sz = to_multi_sz(HARDWARE_ID);
                let hw_id_bytes = std::slice::from_raw_parts(
                    hw_id_multi_sz.as_ptr() as *const u8,
                    hw_id_multi_sz.len() * 2,
                );
                let result = SetupDiSetDeviceRegistryPropertyW(
                    dev_info_set,
                    &mut dev_info_data,
                    SPDRP_HARDWAREID,
                    hw_id_bytes.as_ptr(),
                    hw_id_bytes.len() as u32,
                );
                if result == 0 {
                    warn!(
                        "SetupDiSetDeviceRegistryPropertyW(HARDWAREID) failed (error {})",
                        std::io::Error::last_os_error()
                    );
                    SetupDiDestroyDeviceInfoList(dev_info_set);
                    return None;
                }

                // Set the friendly name (SPDRP_FRIENDLYNAME)
                let friendly_wide = to_wide_null(friendly_name);
                let friendly_bytes = std::slice::from_raw_parts(
                    friendly_wide.as_ptr() as *const u8,
                    friendly_wide.len() * 2,
                );
                let result = SetupDiSetDeviceRegistryPropertyW(
                    dev_info_set,
                    &mut dev_info_data,
                    SPDRP_FRIENDLYNAME,
                    friendly_bytes.as_ptr(),
                    friendly_bytes.len() as u32,
                );
                if result == 0 {
                    warn!(
                        "SetupDiSetDeviceRegistryPropertyW(FRIENDLYNAME) failed (error {})",
                        std::io::Error::last_os_error()
                    );
                    // Non-fatal: continue even if we can't set the friendly name
                }

                // Register the device with DIF_REGISTERDEVICE
                let result = SetupDiCallClassInstaller(
                    DIF_REGISTERDEVICE,
                    dev_info_set,
                    &mut dev_info_data,
                );
                if result == 0 {
                    warn!(
                        "SetupDiCallClassInstaller(DIF_REGISTERDEVICE) failed (error {}); \
                         virtual device not registered",
                        std::io::Error::last_os_error()
                    );
                    SetupDiDestroyDeviceInfoList(dev_info_set);
                    return None;
                }

                // Extract the device instance ID for later removal
                let instance_id = get_device_instance_id(&dev_info_data);

                SetupDiDestroyDeviceInfoList(dev_info_set);

                info!(
                    "registered virtual GPU device: {} (instance: {:?})",
                    friendly_name, instance_id
                );

                Some(VirtualGpuDevice {
                    name: friendly_name.to_string(),
                    index,
                    total_memory,
                    device_instance_id: instance_id,
                })
            }
        }

        /// Remove a virtual GPU device node by index.
        pub(super) fn remove_device(&self, index: u32) {
            let device = match self.devices.iter().find(|d| d.index == index) {
                Some(d) => d,
                None => return,
            };

            let instance_id = match &device.device_instance_id {
                Some(id) => id.clone(),
                None => {
                    warn!(
                        "no device instance ID for device index {}; cannot remove",
                        index
                    );
                    return;
                }
            };

            unsafe {
                // Find and remove the device by enumerating GUID_DEVCLASS_DISPLAY devices
                let dev_info_set = SetupDiGetClassDevsW(
                    &GUID_DEVCLASS_DISPLAY as *const _,
                    std::ptr::null(),
                    std::ptr::null_mut(), // hwndParent
                    DIGCF_PRESENT | DIGCF_PROFILE,
                );
                if dev_info_set == INVALID_HDEVINFO {
                    warn!(
                        "SetupDiGetClassDevsW failed (error {}); cannot remove device",
                        std::io::Error::last_os_error()
                    );
                    return;
                }

                let mut member_index: u32 = 0;
                loop {
                    let mut dev_info_data = SP_DEVINFO_DATA {
                        cbSize: std::mem::size_of::<SP_DEVINFO_DATA>() as u32,
                        ClassGuid: GUID_DEVCLASS_DISPLAY,
                        DevInst: 0,
                        Reserved: 0,
                    };

                    if SetupDiEnumDeviceInfo(dev_info_set, member_index, &mut dev_info_data) == 0 {
                        break; // No more devices
                    }

                    // Check if this is our device by comparing hardware ID
                    if let Some(hw_id) = get_device_registry_string(
                        dev_info_set,
                        &mut dev_info_data,
                        SPDRP_HARDWAREID,
                    ) {
                        if hw_id.contains(HARDWARE_ID) {
                            // Also verify instance ID if we have one
                            if let Some(cur_id) = get_device_instance_id(&dev_info_data) {
                                if cur_id == instance_id {
                                    let result = SetupDiCallClassInstaller(
                                        DIF_REMOVE,
                                        dev_info_set,
                                        &mut dev_info_data,
                                    );
                                    if result == 0 {
                                        // Fallback: try SetupDiRemoveDevice
                                        let result2 = SetupDiRemoveDevice(
                                            dev_info_set,
                                            &mut dev_info_data,
                                        );
                                        if result2 == 0 {
                                            warn!(
                                                "failed to remove virtual device {} (error {})",
                                                instance_id,
                                                std::io::Error::last_os_error()
                                            );
                                        } else {
                                            info!("removed virtual GPU device: {}", instance_id);
                                        }
                                    } else {
                                        info!("removed virtual GPU device: {}", instance_id);
                                    }
                                    break;
                                }
                            }
                        }
                    }

                    member_index += 1;
                }

                SetupDiDestroyDeviceInfoList(dev_info_set);
            }
        }
    }

    /// Read a string registry property from a device info set.
    /// HDEVINFO is `isize` in windows-sys.
    unsafe fn get_device_registry_string(
        dev_info_set: isize,
        dev_info_data: &mut SP_DEVINFO_DATA,
        property: u32,
    ) -> Option<String> {
        let mut buf = [0u16; 512];
        let mut required_size: u32 = 0;
        let mut reg_type: u32 = 0;
        let result = SetupDiGetDeviceRegistryPropertyW(
            dev_info_set,
            dev_info_data,
            property,
            &mut reg_type,
            buf.as_mut_ptr() as *mut u8,
            (buf.len() * 2) as u32,
            &mut required_size,
        );
        if result == 0 {
            return None;
        }
        // Find the first null terminator
        let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        Some(String::from_utf16_lossy(&buf[..len]))
    }

    /// Get the device instance ID string for a device info data entry.
    /// Uses CM_Get_Device_ID_Size_Ex and CM_Get_Device_IDW via the DevInst handle.
    unsafe fn get_device_instance_id(dev_info_data: &SP_DEVINFO_DATA) -> Option<String> {
        let mut id_len: u32 = 0;
        let cr = CM_Get_Device_ID_Size_Ex(
            &mut id_len,
            dev_info_data.DevInst,
            0,
            0, // hmachine: HMACHINE (isize), 0 = local machine
        );
        if cr != 0 {
            return None;
        }

        let mut buf = vec![0u16; (id_len + 1) as usize];
        let cr = CM_Get_Device_IDW(
            dev_info_data.DevInst,
            buf.as_mut_ptr(),
            buf.len() as u32,
            0,
        );
        if cr != 0 {
            return None;
        }

        let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        Some(String::from_utf16_lossy(&buf[..len]))
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
        // (create_device will fail in test env anyway since no driver/module loaded,
        //  but the filtering itself should work)
        dm.sync_devices(&[local_gpu]);

        // No devices should have been added because local GPUs are filtered out
        // (and even if they weren't, create_device would fail in the test environment)
        assert!(dm.devices.is_empty());
    }

    #[test]
    fn test_remove_all_clears_devices() {
        let mut dm = DeviceManager::new();
        dm.remove_all();
        assert!(dm.devices.is_empty());
    }
}
