use tracing::{info, warn};

use rgpu_protocol::gpu_info::{GpuDeviceType, GpuInfo, MemoryHeapInfo};

/// Discover all available GPUs on this machine.
/// Uses Vulkan (via ash) for device enumeration.
pub fn discover_gpus(server_id: u16) -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    // Try Vulkan discovery
    match discover_vulkan_gpus(server_id) {
        Ok(vk_gpus) => {
            info!("discovered {} GPU(s) via Vulkan", vk_gpus.len());
            gpus.extend(vk_gpus);
        }
        Err(e) => {
            warn!("Vulkan GPU discovery failed: {}", e);
        }
    }

    // Fall back to CUDA discovery if Vulkan found nothing
    if gpus.is_empty() {
        match discover_cuda_gpus(server_id) {
            Ok(cuda_gpus) => {
                info!("discovered {} GPU(s) via CUDA fallback", cuda_gpus.len());
                gpus.extend(cuda_gpus);
            }
            Err(e) => {
                warn!("CUDA GPU discovery also failed: {}", e);
            }
        }
    }

    if gpus.is_empty() {
        warn!("no GPUs discovered on this machine");
    }

    gpus
}

/// Fallback GPU discovery using the CUDA driver API.
/// Used when Vulkan is not available (e.g. Docker/WSL2 with GPU passthrough).
fn discover_cuda_gpus(server_id: u16) -> Result<Vec<GpuInfo>, Box<dyn std::error::Error>> {
    use crate::cuda_driver::CudaDriver;

    let driver = CudaDriver::load().map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    let ret = driver.init(0);
    if ret != 0 {
        return Err(format!("cuInit failed: {}", ret).into());
    }

    let count = driver.device_get_count().map_err(|e| format!("cuDeviceGetCount failed: {}", e))?;

    let mut gpus = Vec::new();

    for idx in 0..count {
        let dev = match driver.device_get(idx) {
            Ok(d) => d,
            Err(e) => {
                warn!("cuDeviceGet({}) failed: {}", idx, e);
                continue;
            }
        };

        let device_name = driver.device_get_name(dev)
            .unwrap_or_else(|_| format!("CUDA Device {}", idx));

        let total_memory = driver.device_total_mem(dev)
            .map(|m| m as u64)
            .unwrap_or(0);

        let (major, minor) = driver.device_compute_capability(dev)
            .unwrap_or((0, 0));

        let cuda_compute_capability = if major > 0 {
            Some((major, minor))
        } else {
            None
        };

        let gpu = GpuInfo {
            device_name: device_name.clone(),
            vendor_id: 0x10DE, // NVIDIA
            device_id: 0,
            device_type: GpuDeviceType::DiscreteGpu,
            total_memory,
            supports_vulkan: false,
            supports_cuda: true,
            vulkan_api_version: None,
            vulkan_driver_version: None,
            cuda_compute_capability,
            queue_family_count: 0,
            memory_heaps: vec![MemoryHeapInfo {
                size: total_memory,
                is_device_local: true,
            }],
            server_device_index: idx as u32,
            server_id,
        };

        info!(
            "GPU {} (CUDA): {} ({} MB VRAM, CC {}.{})",
            idx,
            device_name,
            total_memory / (1024 * 1024),
            major,
            minor,
        );

        gpus.push(gpu);
    }

    Ok(gpus)
}

fn discover_vulkan_gpus(server_id: u16) -> Result<Vec<GpuInfo>, Box<dyn std::error::Error>> {
    let entry = unsafe { ash::Entry::load()? };

    let app_info = ash::vk::ApplicationInfo::default()
        .application_name(c"RGPU Server")
        .application_version(ash::vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"RGPU")
        .engine_version(ash::vk::make_api_version(0, 0, 1, 0))
        .api_version(ash::vk::make_api_version(0, 1, 3, 0));

    let create_info = ash::vk::InstanceCreateInfo::default().application_info(&app_info);

    let instance = unsafe { entry.create_instance(&create_info, None)? };

    let physical_devices = unsafe { instance.enumerate_physical_devices()? };

    let mut gpus = Vec::new();

    for (idx, &pd) in physical_devices.iter().enumerate() {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pd) };
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(pd) };

        let device_name = unsafe {
            std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        let device_type = match props.device_type {
            ash::vk::PhysicalDeviceType::DISCRETE_GPU => GpuDeviceType::DiscreteGpu,
            ash::vk::PhysicalDeviceType::INTEGRATED_GPU => GpuDeviceType::IntegratedGpu,
            ash::vk::PhysicalDeviceType::VIRTUAL_GPU => GpuDeviceType::VirtualGpu,
            ash::vk::PhysicalDeviceType::CPU => GpuDeviceType::Cpu,
            _ => GpuDeviceType::Other,
        };

        let memory_heaps: Vec<MemoryHeapInfo> = (0..mem_props.memory_heap_count as usize)
            .map(|i| {
                let heap = mem_props.memory_heaps[i];
                MemoryHeapInfo {
                    size: heap.size,
                    is_device_local: heap
                        .flags
                        .contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL),
                }
            })
            .collect();

        let total_memory = memory_heaps
            .iter()
            .filter(|h| h.is_device_local)
            .map(|h| h.size)
            .sum::<u64>();

        // Check if it's an NVIDIA GPU (CUDA likely supported)
        let is_nvidia = props.vendor_id == 0x10DE;

        let gpu = GpuInfo {
            device_name,
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            device_type,
            total_memory,
            supports_vulkan: true,
            supports_cuda: is_nvidia,
            vulkan_api_version: Some(props.api_version),
            vulkan_driver_version: Some(props.driver_version),
            cuda_compute_capability: None, // Will be filled by CUDA discovery if available
            queue_family_count: queue_families.len() as u32,
            memory_heaps,
            server_device_index: idx as u32,
            server_id,
        };

        info!(
            "GPU {}: {} ({:?}, {}MB VRAM, CUDA: {})",
            idx,
            gpu.device_name,
            gpu.device_type,
            gpu.total_memory / (1024 * 1024),
            gpu.supports_cuda,
        );

        gpus.push(gpu);
    }

    unsafe {
        instance.destroy_instance(None);
    }

    Ok(gpus)
}
