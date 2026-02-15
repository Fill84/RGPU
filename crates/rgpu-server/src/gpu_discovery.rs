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

    if gpus.is_empty() {
        warn!("no GPUs discovered on this machine");
    }

    gpus
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
