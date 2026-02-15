//! Shader module, descriptor set layout, pipeline layout, and compute pipeline functions.

use ash::vk;
use ash::vk::Handle;

use crate::dispatch::DispatchableHandle;
use crate::handle_store;
use crate::send_vulkan_command;

use rgpu_protocol::vulkan_commands::{
    SerializedComputePipelineCreateInfo, SerializedDescriptorSetLayoutBinding,
    SerializedPipelineShaderStageCreateInfo, SerializedPushConstantRange, VulkanCommand,
    VulkanResponse,
};

// ── Shader Module ───────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkCreateShaderModule(
    device: vk::Device,
    p_create_info: *const vk::ShaderModuleCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_shader_module: *mut vk::ShaderModule,
) -> vk::Result {
    if p_create_info.is_null() || p_shader_module.is_null() {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let ci = &*p_create_info;
    let code = std::slice::from_raw_parts(
        ci.p_code as *const u8,
        ci.code_size,
    )
    .to_vec();

    let cmd = VulkanCommand::CreateShaderModule {
        device: dev_handle,
        code,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::ShaderModuleCreated { handle }) => {
            let local_id = handle_store::store_shader_module(handle);
            *p_shader_module = vk::ShaderModule::from_raw(local_id);
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkDestroyShaderModule(
    device: vk::Device,
    shader_module: vk::ShaderModule,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if shader_module == vk::ShaderModule::null() {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let local_id = shader_module.as_raw();
    if let Some(handle) = handle_store::remove_shader_module(local_id) {
        let _ = send_vulkan_command(VulkanCommand::DestroyShaderModule {
            device: dev_handle,
            shader_module: handle,
        });
    }
}

// ── Descriptor Set Layout ───────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkCreateDescriptorSetLayout(
    device: vk::Device,
    p_create_info: *const vk::DescriptorSetLayoutCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_set_layout: *mut vk::DescriptorSetLayout,
) -> vk::Result {
    if p_create_info.is_null() || p_set_layout.is_null() {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let ci = &*p_create_info;
    let mut bindings = Vec::new();
    if !ci.p_bindings.is_null() {
        for i in 0..ci.binding_count as usize {
            let b = &*ci.p_bindings.add(i);
            bindings.push(SerializedDescriptorSetLayoutBinding {
                binding: b.binding,
                descriptor_type: b.descriptor_type.as_raw(),
                descriptor_count: b.descriptor_count,
                stage_flags: b.stage_flags.as_raw(),
            });
        }
    }

    let cmd = VulkanCommand::CreateDescriptorSetLayout {
        device: dev_handle,
        bindings,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::DescriptorSetLayoutCreated { handle }) => {
            let local_id = handle_store::store_desc_set_layout(handle);
            *p_set_layout = vk::DescriptorSetLayout::from_raw(local_id);
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkDestroyDescriptorSetLayout(
    device: vk::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if descriptor_set_layout == vk::DescriptorSetLayout::null() {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let local_id = descriptor_set_layout.as_raw();
    if let Some(handle) = handle_store::remove_desc_set_layout(local_id) {
        let _ = send_vulkan_command(VulkanCommand::DestroyDescriptorSetLayout {
            device: dev_handle,
            layout: handle,
        });
    }
}

// ── Pipeline Layout ─────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkCreatePipelineLayout(
    device: vk::Device,
    p_create_info: *const vk::PipelineLayoutCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_pipeline_layout: *mut vk::PipelineLayout,
) -> vk::Result {
    if p_create_info.is_null() || p_pipeline_layout.is_null() {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let ci = &*p_create_info;

    // Resolve set layout handles
    let mut set_layouts = Vec::new();
    if !ci.p_set_layouts.is_null() {
        for i in 0..ci.set_layout_count as usize {
            let sl = *ci.p_set_layouts.add(i);
            let local_id = sl.as_raw();
            match handle_store::get_desc_set_layout(local_id) {
                Some(h) => set_layouts.push(h),
                None => return vk::Result::ERROR_UNKNOWN,
            }
        }
    }

    // Push constant ranges
    let mut push_constant_ranges = Vec::new();
    if !ci.p_push_constant_ranges.is_null() {
        for i in 0..ci.push_constant_range_count as usize {
            let pcr = &*ci.p_push_constant_ranges.add(i);
            push_constant_ranges.push(SerializedPushConstantRange {
                stage_flags: pcr.stage_flags.as_raw(),
                offset: pcr.offset,
                size: pcr.size,
            });
        }
    }

    let cmd = VulkanCommand::CreatePipelineLayout {
        device: dev_handle,
        set_layouts,
        push_constant_ranges,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::PipelineLayoutCreated { handle }) => {
            let local_id = handle_store::store_pipeline_layout(handle);
            *p_pipeline_layout = vk::PipelineLayout::from_raw(local_id);
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkDestroyPipelineLayout(
    device: vk::Device,
    pipeline_layout: vk::PipelineLayout,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if pipeline_layout == vk::PipelineLayout::null() {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let local_id = pipeline_layout.as_raw();
    if let Some(handle) = handle_store::remove_pipeline_layout(local_id) {
        let _ = send_vulkan_command(VulkanCommand::DestroyPipelineLayout {
            device: dev_handle,
            layout: handle,
        });
    }
}

// ── Compute Pipelines ───────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkCreateComputePipelines(
    device: vk::Device,
    _pipeline_cache: vk::PipelineCache,
    create_info_count: u32,
    p_create_infos: *const vk::ComputePipelineCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_pipelines: *mut vk::Pipeline,
) -> vk::Result {
    if p_create_infos.is_null() || p_pipelines.is_null() || create_info_count == 0 {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let mut create_infos = Vec::new();
    for i in 0..create_info_count as usize {
        let ci = &*p_create_infos.add(i);

        // Resolve shader module handle
        let shader_local_id = ci.stage.module.as_raw();
        let shader_handle = match handle_store::get_shader_module(shader_local_id) {
            Some(h) => h,
            None => return vk::Result::ERROR_UNKNOWN,
        };

        // Read entry point name
        let entry_point = if !ci.stage.p_name.is_null() {
            std::ffi::CStr::from_ptr(ci.stage.p_name)
                .to_string_lossy()
                .into_owned()
        } else {
            "main".to_string()
        };

        // Resolve pipeline layout handle
        let layout_local_id = ci.layout.as_raw();
        let layout_handle = match handle_store::get_pipeline_layout(layout_local_id) {
            Some(h) => h,
            None => return vk::Result::ERROR_UNKNOWN,
        };

        create_infos.push(SerializedComputePipelineCreateInfo {
            stage: SerializedPipelineShaderStageCreateInfo {
                module: shader_handle,
                entry_point,
                stage: ci.stage.stage.as_raw(),
            },
            layout: layout_handle,
            flags: ci.flags.as_raw(),
        });
    }

    let cmd = VulkanCommand::CreateComputePipelines {
        device: dev_handle,
        create_infos,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::PipelinesCreated { handles }) => {
            for (i, h) in handles.iter().enumerate() {
                let local_id = handle_store::store_pipeline(*h);
                *p_pipelines.add(i) = vk::Pipeline::from_raw(local_id);
            }
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkDestroyPipeline(
    device: vk::Device,
    pipeline: vk::Pipeline,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if pipeline == vk::Pipeline::null() {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let local_id = pipeline.as_raw();
    if let Some(handle) = handle_store::remove_pipeline(local_id) {
        let _ = send_vulkan_command(VulkanCommand::DestroyPipeline {
            device: dev_handle,
            pipeline: handle,
        });
    }
}
