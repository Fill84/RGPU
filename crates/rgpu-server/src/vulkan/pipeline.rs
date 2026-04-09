use ash::vk;
use tracing::debug;

use rgpu_protocol::handle::ResourceType;
use rgpu_protocol::vulkan_commands::*;

use crate::session::Session;
use super::VulkanExecutor;

impl VulkanExecutor {
    pub(crate) fn handle_create_shader_module(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, code) = match cmd {
            VulkanCommand::CreateShaderModule { device, code } => (device, code),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid device handle".to_string(),
                }
            }
        };

        // SPIR-V code must be aligned to 4 bytes and size must be multiple of 4
        if code.len() % 4 != 0 {
            return VulkanResponse::Error {
                code: vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                message: "SPIR-V code size must be multiple of 4".to_string(),
            };
        }

        let code_u32: &[u32] = unsafe {
            std::slice::from_raw_parts(
                code.as_ptr() as *const u32,
                code.len() / 4,
            )
        };

        let create_info = vk::ShaderModuleCreateInfo::default().code(code_u32);
        match unsafe { dev.create_shader_module(&create_info, None) } {
            Ok(module) => {
                let handle = session.alloc_handle(ResourceType::VkShaderModule);
                self.shader_module_handles.insert(handle, module);
                self.shader_to_device.insert(handle, device);
                debug!("created shader module: {:?}", handle);
                VulkanResponse::ShaderModuleCreated { handle }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_destroy_shader_module(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, shader_module) = match cmd {
            VulkanCommand::DestroyShaderModule {
                device,
                shader_module,
            } => (device, shader_module),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };
        if let Some((_, module)) = self.shader_module_handles.remove(&shader_module) {
            unsafe { dev.destroy_shader_module(module, None) };
            self.shader_to_device.remove(&shader_module);
            session.remove_handle(&shader_module);
        }
        VulkanResponse::Success
    }

    pub(crate) fn handle_create_descriptor_set_layout(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, bindings) = match cmd {
            VulkanCommand::CreateDescriptorSetLayout { device, bindings } => (device, bindings),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid device handle".to_string(),
                }
            }
        };

        let vk_bindings: Vec<vk::DescriptorSetLayoutBinding> = bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(vk::DescriptorType::from_raw(b.descriptor_type))
                    .descriptor_count(b.descriptor_count)
                    .stage_flags(vk::ShaderStageFlags::from_raw(b.stage_flags))
            })
            .collect();

        let create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_bindings);

        match unsafe { dev.create_descriptor_set_layout(&create_info, None) } {
            Ok(layout) => {
                let handle = session.alloc_handle(ResourceType::VkDescriptorSetLayout);
                self.desc_set_layout_handles.insert(handle, layout);
                self.desc_set_layout_to_device.insert(handle, device);
                VulkanResponse::DescriptorSetLayoutCreated { handle }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_destroy_descriptor_set_layout(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, layout) = match cmd {
            VulkanCommand::DestroyDescriptorSetLayout { device, layout } => (device, layout),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };
        if let Some((_, l)) = self.desc_set_layout_handles.remove(&layout) {
            unsafe { dev.destroy_descriptor_set_layout(l, None) };
            self.desc_set_layout_to_device.remove(&layout);
            session.remove_handle(&layout);
        }
        VulkanResponse::Success
    }

    pub(crate) fn handle_create_pipeline_layout(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, set_layouts, push_constant_ranges) = match cmd {
            VulkanCommand::CreatePipelineLayout {
                device,
                set_layouts,
                push_constant_ranges,
            } => (device, set_layouts, push_constant_ranges),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid device handle".to_string(),
                }
            }
        };

        let vk_layouts: Vec<vk::DescriptorSetLayout> = set_layouts
            .iter()
            .filter_map(|h| {
                self.desc_set_layout_handles
                    .get(h)
                    .map(|v| *v.value())
            })
            .collect();

        let vk_push_ranges: Vec<vk::PushConstantRange> = push_constant_ranges
            .iter()
            .map(|p| {
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::from_raw(p.stage_flags))
                    .offset(p.offset)
                    .size(p.size)
            })
            .collect();

        let create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&vk_layouts)
            .push_constant_ranges(&vk_push_ranges);

        match unsafe { dev.create_pipeline_layout(&create_info, None) } {
            Ok(layout) => {
                let handle = session.alloc_handle(ResourceType::VkPipelineLayout);
                self.pipeline_layout_handles.insert(handle, layout);
                self.pipeline_layout_to_device.insert(handle, device);
                VulkanResponse::PipelineLayoutCreated { handle }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_destroy_pipeline_layout(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, layout) = match cmd {
            VulkanCommand::DestroyPipelineLayout { device, layout } => (device, layout),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };
        if let Some((_, l)) = self.pipeline_layout_handles.remove(&layout) {
            unsafe { dev.destroy_pipeline_layout(l, None) };
            self.pipeline_layout_to_device.remove(&layout);
            session.remove_handle(&layout);
        }
        VulkanResponse::Success
    }

    pub(crate) fn handle_create_compute_pipelines(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, create_infos) = match cmd {
            VulkanCommand::CreateComputePipelines {
                device,
                create_infos,
            } => (device, create_infos),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid device handle".to_string(),
                }
            }
        };

        let entry_points: Vec<std::ffi::CString> = create_infos
            .iter()
            .map(|ci| {
                std::ffi::CString::new(ci.stage.entry_point.as_str()).unwrap_or_default()
            })
            .collect();

        let vk_create_infos: Vec<vk::ComputePipelineCreateInfo> = create_infos
            .iter()
            .enumerate()
            .map(|(i, ci)| {
                let module = self
                    .shader_module_handles
                    .get(&ci.stage.module)
                    .map(|v| *v.value())
                    .unwrap_or(vk::ShaderModule::null());
                let layout = self
                    .pipeline_layout_handles
                    .get(&ci.layout)
                    .map(|v| *v.value())
                    .unwrap_or(vk::PipelineLayout::null());

                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::from_raw(ci.stage.stage))
                    .module(module)
                    .name(entry_points[i].as_c_str());

                vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)
                    .flags(vk::PipelineCreateFlags::from_raw(ci.flags))
            })
            .collect();

        match unsafe {
            dev.create_compute_pipelines(
                vk::PipelineCache::null(),
                &vk_create_infos,
                None,
            )
        } {
            Ok(pipelines) => {
                let mut handles = Vec::new();
                for pipeline in pipelines {
                    let handle = session.alloc_handle(ResourceType::VkPipeline);
                    self.pipeline_handles.insert(handle, pipeline);
                    self.pipeline_to_device.insert(handle, device);
                    handles.push(handle);
                }
                VulkanResponse::PipelinesCreated { handles }
            }
            Err((_, e)) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_destroy_pipeline(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, pipeline) = match cmd {
            VulkanCommand::DestroyPipeline { device, pipeline } => (device, pipeline),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };
        if let Some((_, p)) = self.pipeline_handles.remove(&pipeline) {
            unsafe { dev.destroy_pipeline(p, None) };
            self.pipeline_to_device.remove(&pipeline);
            session.remove_handle(&pipeline);
        }
        VulkanResponse::Success
    }
}
