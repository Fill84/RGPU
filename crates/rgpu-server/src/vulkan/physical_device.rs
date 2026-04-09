use std::ffi::CStr;

use ash::vk;

use rgpu_protocol::vulkan_commands::*;

use super::VulkanExecutor;

impl VulkanExecutor {
    pub(crate) fn handle_get_physical_device_properties(
        &self,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let physical_device = match cmd {
            VulkanCommand::GetPhysicalDeviceProperties { physical_device }
            | VulkanCommand::GetPhysicalDeviceProperties2 { physical_device } => physical_device,
            _ => unreachable!(),
        };

        let (pd, inst_handle) = match self.physical_device_handles.get(&physical_device) {
            Some(e) => *e.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid physical device handle".to_string(),
                }
            }
        };
        let wrapper = match self.instance_wrappers.get(&inst_handle) {
            Some(w) => w,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                    message: "instance wrapper not found".to_string(),
                }
            }
        };

        let props = unsafe { wrapper.get_physical_device_properties(pd) };

        let device_name = unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        // Serialize limits as raw bytes
        let limits_raw = unsafe {
            let ptr = &props.limits as *const vk::PhysicalDeviceLimits as *const u8;
            std::slice::from_raw_parts(
                ptr,
                std::mem::size_of::<vk::PhysicalDeviceLimits>(),
            )
            .to_vec()
        };

        let sparse_properties_raw = unsafe {
            let ptr = &props.sparse_properties
                as *const vk::PhysicalDeviceSparseProperties
                as *const u8;
            std::slice::from_raw_parts(
                ptr,
                std::mem::size_of::<vk::PhysicalDeviceSparseProperties>(),
            )
            .to_vec()
        };

        VulkanResponse::PhysicalDeviceProperties {
            api_version: props.api_version,
            driver_version: props.driver_version,
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            device_type: props.device_type.as_raw() as u32,
            device_name,
            pipeline_cache_uuid: props.pipeline_cache_uuid,
            limits_raw,
            sparse_properties_raw,
        }
    }

    pub(crate) fn handle_get_physical_device_features(
        &self,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let physical_device = match cmd {
            VulkanCommand::GetPhysicalDeviceFeatures { physical_device }
            | VulkanCommand::GetPhysicalDeviceFeatures2 { physical_device } => physical_device,
            _ => unreachable!(),
        };

        let (pd, inst_handle) = match self.physical_device_handles.get(&physical_device) {
            Some(e) => *e.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid physical device handle".to_string(),
                }
            }
        };
        let wrapper = match self.instance_wrappers.get(&inst_handle) {
            Some(w) => w,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                    message: "instance wrapper not found".to_string(),
                }
            }
        };

        let features = unsafe { wrapper.get_physical_device_features(pd) };
        let features_raw = unsafe {
            let ptr = &features as *const vk::PhysicalDeviceFeatures as *const u8;
            std::slice::from_raw_parts(
                ptr,
                std::mem::size_of::<vk::PhysicalDeviceFeatures>(),
            )
            .to_vec()
        };

        VulkanResponse::PhysicalDeviceFeatures { features_raw }
    }

    pub(crate) fn handle_get_physical_device_memory_properties(
        &self,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let physical_device = match cmd {
            VulkanCommand::GetPhysicalDeviceMemoryProperties { physical_device }
            | VulkanCommand::GetPhysicalDeviceMemoryProperties2 { physical_device } => {
                physical_device
            }
            _ => unreachable!(),
        };

        let (pd, inst_handle) = match self.physical_device_handles.get(&physical_device) {
            Some(e) => *e.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid physical device handle".to_string(),
                }
            }
        };
        let wrapper = match self.instance_wrappers.get(&inst_handle) {
            Some(w) => w,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                    message: "instance wrapper not found".to_string(),
                }
            }
        };

        let mem_props =
            unsafe { wrapper.get_physical_device_memory_properties(pd) };

        let memory_types: Vec<SerializedMemoryType> =
            (0..mem_props.memory_type_count as usize)
                .map(|i| SerializedMemoryType {
                    property_flags: mem_props.memory_types[i].property_flags.as_raw(),
                    heap_index: mem_props.memory_types[i].heap_index,
                })
                .collect();

        let memory_heaps: Vec<SerializedMemoryHeap> =
            (0..mem_props.memory_heap_count as usize)
                .map(|i| SerializedMemoryHeap {
                    size: mem_props.memory_heaps[i].size,
                    flags: mem_props.memory_heaps[i].flags.as_raw(),
                })
                .collect();

        VulkanResponse::PhysicalDeviceMemoryProperties {
            memory_type_count: mem_props.memory_type_count,
            memory_types,
            memory_heap_count: mem_props.memory_heap_count,
            memory_heaps,
        }
    }

    pub(crate) fn handle_get_physical_device_queue_family_properties(
        &self,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let physical_device = match cmd {
            VulkanCommand::GetPhysicalDeviceQueueFamilyProperties { physical_device }
            | VulkanCommand::GetPhysicalDeviceQueueFamilyProperties2 { physical_device } => {
                physical_device
            }
            _ => unreachable!(),
        };

        let (pd, inst_handle) = match self.physical_device_handles.get(&physical_device) {
            Some(e) => *e.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid physical device handle".to_string(),
                }
            }
        };
        let wrapper = match self.instance_wrappers.get(&inst_handle) {
            Some(w) => w,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                    message: "instance wrapper not found".to_string(),
                }
            }
        };

        let queue_families =
            unsafe { wrapper.get_physical_device_queue_family_properties(pd) };

        let families: Vec<SerializedQueueFamilyProperties> = queue_families
            .iter()
            .map(|qf| SerializedQueueFamilyProperties {
                queue_flags: qf.queue_flags.as_raw(),
                queue_count: qf.queue_count,
                timestamp_valid_bits: qf.timestamp_valid_bits,
                min_image_transfer_granularity: [
                    qf.min_image_transfer_granularity.width,
                    qf.min_image_transfer_granularity.height,
                    qf.min_image_transfer_granularity.depth,
                ],
            })
            .collect();

        VulkanResponse::QueueFamilyProperties { families }
    }

    pub(crate) fn handle_get_physical_device_format_properties(
        &self,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (physical_device, format) = match cmd {
            VulkanCommand::GetPhysicalDeviceFormatProperties {
                physical_device,
                format,
            } => (physical_device, format),
            _ => unreachable!(),
        };

        let (pd, inst_handle) = match self.physical_device_handles.get(&physical_device) {
            Some(e) => *e.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid physical device handle".to_string(),
                }
            }
        };
        let wrapper = match self.instance_wrappers.get(&inst_handle) {
            Some(w) => w,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                    message: "instance wrapper not found".to_string(),
                }
            }
        };

        let vk_format = vk::Format::from_raw(format);
        let format_props =
            unsafe { wrapper.get_physical_device_format_properties(pd, vk_format) };

        VulkanResponse::FormatProperties {
            linear_tiling_features: format_props.linear_tiling_features.as_raw(),
            optimal_tiling_features: format_props.optimal_tiling_features.as_raw(),
            buffer_features: format_props.buffer_features.as_raw(),
        }
    }
}
