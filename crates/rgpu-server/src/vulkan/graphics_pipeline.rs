use ash::vk;
use tracing::debug;

use rgpu_protocol::handle::ResourceType;
use rgpu_protocol::vulkan_commands::*;

use crate::session::Session;
use super::VulkanExecutor;

impl VulkanExecutor {
    pub(crate) fn handle_create_graphics_pipelines(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, create_infos) = match cmd {
            VulkanCommand::CreateGraphicsPipelines {
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

        // Build all pipeline create infos - careful with lifetimes
        let mut vk_create_infos: Vec<vk::GraphicsPipelineCreateInfo> = Vec::new();

        // Keep all intermediate data alive
        let mut all_stages: Vec<Vec<vk::PipelineShaderStageCreateInfo>> = Vec::new();
        let mut all_entry_points: Vec<Vec<std::ffi::CString>> = Vec::new();
        let mut all_vi_bindings: Vec<Vec<vk::VertexInputBindingDescription>> = Vec::new();
        let mut all_vi_attrs: Vec<Vec<vk::VertexInputAttributeDescription>> = Vec::new();
        let mut all_vi_states: Vec<vk::PipelineVertexInputStateCreateInfo> = Vec::new();
        let mut all_ia_states: Vec<vk::PipelineInputAssemblyStateCreateInfo> = Vec::new();
        let mut all_viewports: Vec<Vec<vk::Viewport>> = Vec::new();
        let mut all_scissors: Vec<Vec<vk::Rect2D>> = Vec::new();
        let mut all_vp_states: Vec<vk::PipelineViewportStateCreateInfo> = Vec::new();
        let mut all_rs_states: Vec<vk::PipelineRasterizationStateCreateInfo> = Vec::new();
        let mut all_ms_states: Vec<vk::PipelineMultisampleStateCreateInfo> = Vec::new();
        let mut all_ds_states: Vec<vk::PipelineDepthStencilStateCreateInfo> = Vec::new();
        let mut all_cb_attachments: Vec<Vec<vk::PipelineColorBlendAttachmentState>> =
            Vec::new();
        let mut all_cb_states: Vec<vk::PipelineColorBlendStateCreateInfo> = Vec::new();
        let mut all_dyn_states_raw: Vec<Vec<vk::DynamicState>> = Vec::new();
        let mut all_dyn_states: Vec<vk::PipelineDynamicStateCreateInfo> = Vec::new();

        // Phase 1: Collect all raw data into Vecs (no references created yet)
        for ci in &create_infos {
            // Shader stages - collect entry points
            let mut entry_points = Vec::new();
            for stage in &ci.stages {
                if self.shader_module_handles.get(&stage.module).is_none() {
                    return VulkanResponse::Error {
                        code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                        message: "invalid shader module handle".to_string(),
                    };
                }
                let ep =
                    std::ffi::CString::new(stage.entry_point.as_str()).unwrap_or_default();
                entry_points.push(ep);
            }
            all_entry_points.push(entry_points);

            // Vertex input
            let vi_bindings: Vec<vk::VertexInputBindingDescription> = ci
                .vertex_input_state
                .vertex_binding_descriptions
                .iter()
                .map(|b| vk::VertexInputBindingDescription {
                    binding: b.binding,
                    stride: b.stride,
                    input_rate: vk::VertexInputRate::from_raw(b.input_rate),
                })
                .collect();
            let vi_attrs: Vec<vk::VertexInputAttributeDescription> = ci
                .vertex_input_state
                .vertex_attribute_descriptions
                .iter()
                .map(|a| vk::VertexInputAttributeDescription {
                    location: a.location,
                    binding: a.binding,
                    format: vk::Format::from_raw(a.format),
                    offset: a.offset,
                })
                .collect();
            all_vi_bindings.push(vi_bindings);
            all_vi_attrs.push(vi_attrs);

            // Viewport/scissor data
            if let Some(ref vps) = ci.viewport_state {
                let viewports: Vec<vk::Viewport> = vps
                    .viewports
                    .iter()
                    .map(|v| vk::Viewport {
                        x: v.x,
                        y: v.y,
                        width: v.width,
                        height: v.height,
                        min_depth: v.min_depth,
                        max_depth: v.max_depth,
                    })
                    .collect();
                let scissors: Vec<vk::Rect2D> = vps
                    .scissors
                    .iter()
                    .map(|s| vk::Rect2D {
                        offset: vk::Offset2D {
                            x: s.offset[0],
                            y: s.offset[1],
                        },
                        extent: vk::Extent2D {
                            width: s.extent[0],
                            height: s.extent[1],
                        },
                    })
                    .collect();
                all_viewports.push(viewports);
                all_scissors.push(scissors);
            } else {
                all_viewports.push(Vec::new());
                all_scissors.push(Vec::new());
            }

            // Color blend attachments
            if let Some(ref cbs) = ci.color_blend_state {
                let blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = cbs
                    .attachments
                    .iter()
                    .map(|a| {
                        vk::PipelineColorBlendAttachmentState::default()
                            .blend_enable(a.blend_enable)
                            .src_color_blend_factor(vk::BlendFactor::from_raw(
                                a.src_color_blend_factor,
                            ))
                            .dst_color_blend_factor(vk::BlendFactor::from_raw(
                                a.dst_color_blend_factor,
                            ))
                            .color_blend_op(vk::BlendOp::from_raw(a.color_blend_op))
                            .src_alpha_blend_factor(vk::BlendFactor::from_raw(
                                a.src_alpha_blend_factor,
                            ))
                            .dst_alpha_blend_factor(vk::BlendFactor::from_raw(
                                a.dst_alpha_blend_factor,
                            ))
                            .alpha_blend_op(vk::BlendOp::from_raw(a.alpha_blend_op))
                            .color_write_mask(vk::ColorComponentFlags::from_raw(
                                a.color_write_mask,
                            ))
                    })
                    .collect();
                all_cb_attachments.push(blend_attachments);
            } else {
                all_cb_attachments.push(Vec::new());
            }

            // Dynamic states
            if let Some(ref dyn_s) = ci.dynamic_state {
                let dyn_states: Vec<vk::DynamicState> = dyn_s
                    .dynamic_states
                    .iter()
                    .map(|d| vk::DynamicState::from_raw(*d))
                    .collect();
                all_dyn_states_raw.push(dyn_states);
            } else {
                all_dyn_states_raw.push(Vec::new());
            }
        }

        // Phase 2: Build all CreateInfo structs (now all Vecs are stable, no more pushes)
        for (i, ci) in create_infos.iter().enumerate() {
            // Shader stages
            let mut stages = Vec::new();
            for (j, stage) in ci.stages.iter().enumerate() {
                let module = match self.shader_module_handles.get(&stage.module) {
                    Some(m) => *m.value(),
                    None => {
                        return VulkanResponse::Error {
                            code: ash::vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                            message: format!("shader module handle {:?} not found", stage.module),
                        };
                    }
                };
                stages.push(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::from_raw(stage.stage))
                        .module(module)
                        .name(all_entry_points[i][j].as_c_str()),
                );
            }
            all_stages.push(stages);

            // Vertex input state
            let vi_state = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&all_vi_bindings[i])
                .vertex_attribute_descriptions(&all_vi_attrs[i]);
            all_vi_states.push(vi_state);

            // Input assembly
            let ia_state = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::from_raw(
                    ci.input_assembly_state.topology,
                ))
                .primitive_restart_enable(ci.input_assembly_state.primitive_restart_enable);
            all_ia_states.push(ia_state);

            // Viewport state
            if ci.viewport_state.is_some() {
                let vp_state = vk::PipelineViewportStateCreateInfo::default()
                    .viewports(&all_viewports[i])
                    .scissors(&all_scissors[i]);
                all_vp_states.push(vp_state);
            } else {
                all_vp_states.push(vk::PipelineViewportStateCreateInfo::default());
            }

            // Rasterization state
            let rs = &ci.rasterization_state;
            let rs_state = vk::PipelineRasterizationStateCreateInfo::default()
                .depth_clamp_enable(rs.depth_clamp_enable)
                .rasterizer_discard_enable(rs.rasterizer_discard_enable)
                .polygon_mode(vk::PolygonMode::from_raw(rs.polygon_mode))
                .cull_mode(vk::CullModeFlags::from_raw(rs.cull_mode))
                .front_face(vk::FrontFace::from_raw(rs.front_face))
                .depth_bias_enable(rs.depth_bias_enable)
                .depth_bias_constant_factor(rs.depth_bias_constant_factor)
                .depth_bias_clamp(rs.depth_bias_clamp)
                .depth_bias_slope_factor(rs.depth_bias_slope_factor)
                .line_width(rs.line_width);
            all_rs_states.push(rs_state);

            // Multisample state
            if let Some(ref ms) = ci.multisample_state {
                let ms_state = vk::PipelineMultisampleStateCreateInfo::default()
                    .rasterization_samples(vk::SampleCountFlags::from_raw(
                        ms.rasterization_samples,
                    ))
                    .sample_shading_enable(ms.sample_shading_enable)
                    .min_sample_shading(ms.min_sample_shading)
                    .alpha_to_coverage_enable(ms.alpha_to_coverage_enable)
                    .alpha_to_one_enable(ms.alpha_to_one_enable);
                all_ms_states.push(ms_state);
            } else {
                all_ms_states.push(
                    vk::PipelineMultisampleStateCreateInfo::default()
                        .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                );
            }

            // Depth stencil state
            if let Some(ref ds) = ci.depth_stencil_state {
                let ds_state = vk::PipelineDepthStencilStateCreateInfo::default()
                    .depth_test_enable(ds.depth_test_enable)
                    .depth_write_enable(ds.depth_write_enable)
                    .depth_compare_op(vk::CompareOp::from_raw(ds.depth_compare_op))
                    .depth_bounds_test_enable(ds.depth_bounds_test_enable)
                    .stencil_test_enable(ds.stencil_test_enable)
                    .front(vk::StencilOpState {
                        fail_op: vk::StencilOp::from_raw(ds.front.fail_op),
                        pass_op: vk::StencilOp::from_raw(ds.front.pass_op),
                        depth_fail_op: vk::StencilOp::from_raw(ds.front.depth_fail_op),
                        compare_op: vk::CompareOp::from_raw(ds.front.compare_op),
                        compare_mask: ds.front.compare_mask,
                        write_mask: ds.front.write_mask,
                        reference: ds.front.reference,
                    })
                    .back(vk::StencilOpState {
                        fail_op: vk::StencilOp::from_raw(ds.back.fail_op),
                        pass_op: vk::StencilOp::from_raw(ds.back.pass_op),
                        depth_fail_op: vk::StencilOp::from_raw(ds.back.depth_fail_op),
                        compare_op: vk::CompareOp::from_raw(ds.back.compare_op),
                        compare_mask: ds.back.compare_mask,
                        write_mask: ds.back.write_mask,
                        reference: ds.back.reference,
                    })
                    .min_depth_bounds(ds.min_depth_bounds)
                    .max_depth_bounds(ds.max_depth_bounds);
                all_ds_states.push(ds_state);
            } else {
                all_ds_states.push(vk::PipelineDepthStencilStateCreateInfo::default());
            }

            // Color blend state
            if let Some(ref cbs) = ci.color_blend_state {
                let cb_state = vk::PipelineColorBlendStateCreateInfo::default()
                    .logic_op_enable(cbs.logic_op_enable)
                    .logic_op(vk::LogicOp::from_raw(cbs.logic_op))
                    .attachments(&all_cb_attachments[i])
                    .blend_constants(cbs.blend_constants);
                all_cb_states.push(cb_state);
            } else {
                all_cb_states.push(vk::PipelineColorBlendStateCreateInfo::default());
            }

            // Dynamic state
            if ci.dynamic_state.is_some() && !all_dyn_states_raw[i].is_empty() {
                let dyn_state = vk::PipelineDynamicStateCreateInfo::default()
                    .dynamic_states(&all_dyn_states_raw[i]);
                all_dyn_states.push(dyn_state);
            } else {
                all_dyn_states.push(vk::PipelineDynamicStateCreateInfo::default());
            }
        }

        // Phase 3: Assemble final pipeline create infos
        for (i, ci) in create_infos.iter().enumerate() {
            let layout = match self.pipeline_layout_handles.get(&ci.layout) {
                Some(l) => *l.value(),
                None => {
                    return VulkanResponse::Error {
                        code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                        message: "invalid pipeline layout handle".to_string(),
                    }
                }
            };
            let rp = match self.render_pass_handles.get(&ci.render_pass) {
                Some(r) => *r.value(),
                None => {
                    return VulkanResponse::Error {
                        code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                        message: "invalid render pass handle".to_string(),
                    }
                }
            };

            let mut pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
                .flags(vk::PipelineCreateFlags::from_raw(ci.flags))
                .stages(&all_stages[i])
                .vertex_input_state(&all_vi_states[i])
                .input_assembly_state(&all_ia_states[i])
                .viewport_state(&all_vp_states[i])
                .rasterization_state(&all_rs_states[i])
                .multisample_state(&all_ms_states[i])
                .depth_stencil_state(&all_ds_states[i])
                .color_blend_state(&all_cb_states[i])
                .layout(layout)
                .render_pass(rp)
                .subpass(ci.subpass);

            if ci.dynamic_state.is_some()
                && !all_dyn_states_raw[i].is_empty()
            {
                pipeline_ci = pipeline_ci.dynamic_state(&all_dyn_states[i]);
            }

            vk_create_infos.push(pipeline_ci);
        }

        match unsafe {
            dev.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &vk_create_infos,
                None,
            )
        } {
            Ok(pipelines) => {
                let mut handles = Vec::new();
                for p in pipelines {
                    let handle = session.alloc_handle(ResourceType::VkPipeline);
                    self.pipeline_handles.insert(handle, p);
                    self.pipeline_to_device.insert(handle, device);
                    handles.push(handle);
                }
                debug!("created {} graphics pipeline(s)", handles.len());
                VulkanResponse::PipelinesCreated { handles }
            }
            Err((pipelines, e)) => {
                // Some pipelines may have succeeded
                for p in pipelines {
                    if p != vk::Pipeline::null() {
                        unsafe { dev.destroy_pipeline(p, None) };
                    }
                }
                Self::vk_err(e)
            }
        }
    }
}
