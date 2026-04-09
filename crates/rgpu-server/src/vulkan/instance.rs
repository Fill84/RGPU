use std::sync::Arc;

use ash::vk;
use tracing::{debug, info};

use rgpu_protocol::handle::ResourceType;
use rgpu_protocol::vulkan_commands::*;

use crate::session::Session;
use super::VulkanExecutor;

impl VulkanExecutor {
    pub(crate) fn handle_create_instance(
        &self,
        session: &Session,
        entry: &Arc<ash::Entry>,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (
            app_name,
            app_version,
            engine_name,
            engine_version,
            api_version,
        ) = match cmd {
            VulkanCommand::CreateInstance {
                app_name,
                app_version,
                engine_name,
                engine_version,
                api_version,
                enabled_extensions: _,
                enabled_layers: _,
            } => (app_name, app_version, engine_name, engine_version, api_version),
            _ => unreachable!(),
        };

        let app_name_c = app_name
            .as_deref()
            .map(|s| std::ffi::CString::new(s).unwrap_or_default());
        let engine_name_c = engine_name
            .as_deref()
            .map(|s| std::ffi::CString::new(s).unwrap_or_default());

        let mut app_info = vk::ApplicationInfo::default()
            .application_version(app_version)
            .engine_version(engine_version)
            .api_version(if api_version == 0 {
                vk::make_api_version(0, 1, 3, 0)
            } else {
                api_version
            });

        if let Some(ref name) = app_name_c {
            app_info = app_info.application_name(name.as_c_str());
        }
        if let Some(ref name) = engine_name_c {
            app_info = app_info.engine_name(name.as_c_str());
        }

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        match unsafe { entry.create_instance(&create_info, None) } {
            Ok(instance) => {
                let handle = session.alloc_handle(ResourceType::VkInstance);
                let raw = instance.handle();
                self.instance_handles.insert(handle, raw);
                self.instance_wrappers.insert(handle, instance);
                info!("created Vulkan instance: {:?}", handle);
                VulkanResponse::InstanceCreated { handle }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_destroy_instance(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let instance = match cmd {
            VulkanCommand::DestroyInstance { instance } => instance,
            _ => unreachable!(),
        };

        if let Some((_, inst)) = self.instance_handles.remove(&instance) {
            if let Some((_, wrapper)) = self.instance_wrappers.remove(&instance) {
                unsafe { wrapper.destroy_instance(None) };
            }
            // Clean up physical devices belonging to this instance
            let pd_keys: Vec<_> = self
                .physical_device_handles
                .iter()
                .filter(|e| e.value().1 == instance)
                .map(|e| *e.key())
                .collect();
            for key in pd_keys {
                self.physical_device_handles.remove(&key);
            }
            let _ = inst;
            session.remove_handle(&instance);
            debug!("destroyed Vulkan instance: {:?}", instance);
        }
        VulkanResponse::Success
    }

    pub(crate) fn handle_enumerate_physical_devices(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let instance = match cmd {
            VulkanCommand::EnumeratePhysicalDevices { instance } => instance,
            _ => unreachable!(),
        };

        let wrapper = match self.instance_wrappers.get(&instance) {
            Some(w) => w,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                    message: "invalid instance handle".to_string(),
                }
            }
        };

        match unsafe { wrapper.enumerate_physical_devices() } {
            Ok(physical_devices) => {
                let mut handles = Vec::new();
                for pd in physical_devices {
                    let handle = session.alloc_handle(ResourceType::VkPhysicalDevice);
                    self.physical_device_handles
                        .insert(handle, (pd, instance));
                    handles.push(handle);
                }
                debug!("enumerated {} physical devices", handles.len());
                VulkanResponse::PhysicalDevices { handles }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_enumerate_instance_extension_properties(
        &self,
        _cmd: VulkanCommand,
    ) -> VulkanResponse {
        // For Phase 3, report a minimal set of extensions
        let extensions = vec![
            SerializedExtensionProperties {
                extension_name: "VK_KHR_get_physical_device_properties2".to_string(),
                spec_version: 2,
            },
        ];
        VulkanResponse::ExtensionProperties { extensions }
    }

    pub(crate) fn handle_enumerate_instance_layer_properties(&self) -> VulkanResponse {
        VulkanResponse::LayerProperties { layers: vec![] }
    }

    pub(crate) fn handle_enumerate_device_extension_properties(
        &self,
        _cmd: VulkanCommand,
    ) -> VulkanResponse {
        // Minimal device extensions for Phase 3
        VulkanResponse::ExtensionProperties {
            extensions: vec![],
        }
    }
}
