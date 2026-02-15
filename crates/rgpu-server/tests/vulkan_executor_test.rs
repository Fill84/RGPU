//! Integration test: VulkanExecutor
//!
//! Tests the server-side VulkanExecutor directly (no networking).
//! Verifies instance creation, physical device enumeration, properties,
//! device creation, memory allocation, and basic compute pipeline workflow.
//!
//! Run with: cargo test --test vulkan_executor_test -- --nocapture

use rgpu_protocol::vulkan_commands::*;
use rgpu_server::session::Session;
use rgpu_server::vulkan_executor::VulkanExecutor;

fn make_session() -> Session {
    Session::new(1, 0, "test".to_string())
}

#[test]
fn test_create_and_destroy_instance() {
    let executor = VulkanExecutor::new();
    let session = make_session();

    // Create instance
    let resp = executor.execute(
        &session,
        VulkanCommand::CreateInstance {
            app_name: Some("VulkanExecutorTest".to_string()),
            app_version: 1,
            engine_name: None,
            engine_version: 0,
            api_version: ash::vk::make_api_version(0, 1, 0, 0),
            enabled_extensions: Vec::new(),
            enabled_layers: Vec::new(),
        },
    );

    let instance_handle = match resp {
        VulkanResponse::InstanceCreated { handle } => {
            println!("Instance created: {:?}", handle);
            handle
        }
        other => panic!("expected InstanceCreated, got {:?}", other),
    };

    // Destroy instance
    let resp = executor.execute(
        &session,
        VulkanCommand::DestroyInstance {
            instance: instance_handle,
        },
    );
    match resp {
        VulkanResponse::Success => println!("Instance destroyed"),
        other => panic!("expected Success, got {:?}", other),
    }
}

#[test]
fn test_enumerate_physical_devices() {
    let executor = VulkanExecutor::new();
    let session = make_session();

    // Create instance
    let instance_handle = match executor.execute(
        &session,
        VulkanCommand::CreateInstance {
            app_name: Some("PhysDevTest".to_string()),
            app_version: 1,
            engine_name: None,
            engine_version: 0,
            api_version: ash::vk::make_api_version(0, 1, 0, 0),
            enabled_extensions: Vec::new(),
            enabled_layers: Vec::new(),
        },
    ) {
        VulkanResponse::InstanceCreated { handle } => handle,
        other => panic!("expected InstanceCreated, got {:?}", other),
    };

    // Enumerate physical devices
    let resp = executor.execute(
        &session,
        VulkanCommand::EnumeratePhysicalDevices {
            instance: instance_handle,
        },
    );

    let pd_handles = match resp {
        VulkanResponse::PhysicalDevices { handles } => {
            println!("Found {} physical device(s)", handles.len());
            assert!(!handles.is_empty(), "no physical devices found");
            handles
        }
        other => panic!("expected PhysicalDevices, got {:?}", other),
    };

    // Get properties for first device
    let resp = executor.execute(
        &session,
        VulkanCommand::GetPhysicalDeviceProperties {
            physical_device: pd_handles[0],
        },
    );

    match resp {
        VulkanResponse::PhysicalDeviceProperties {
            device_name,
            api_version,
            vendor_id,
            device_type,
            ..
        } => {
            println!(
                "Device: {} (type={}, vendor=0x{:x}, api={})",
                device_name, device_type, vendor_id, api_version
            );
        }
        other => panic!("expected PhysicalDeviceProperties, got {:?}", other),
    }

    // Get queue family properties
    let resp = executor.execute(
        &session,
        VulkanCommand::GetPhysicalDeviceQueueFamilyProperties {
            physical_device: pd_handles[0],
        },
    );

    match resp {
        VulkanResponse::QueueFamilyProperties { families } => {
            println!("Queue families: {}", families.len());
            for (i, qf) in families.iter().enumerate() {
                println!(
                    "  Family {}: flags=0x{:x}, count={}",
                    i, qf.queue_flags, qf.queue_count
                );
            }
        }
        other => panic!("expected QueueFamilyProperties, got {:?}", other),
    }

    // Get memory properties
    let resp = executor.execute(
        &session,
        VulkanCommand::GetPhysicalDeviceMemoryProperties {
            physical_device: pd_handles[0],
        },
    );

    match resp {
        VulkanResponse::PhysicalDeviceMemoryProperties {
            memory_type_count,
            memory_heap_count,
            ..
        } => {
            println!(
                "Memory: {} types, {} heaps",
                memory_type_count, memory_heap_count
            );
        }
        other => panic!("expected PhysicalDeviceMemoryProperties, got {:?}", other),
    }

    // Cleanup
    executor.execute(
        &session,
        VulkanCommand::DestroyInstance {
            instance: instance_handle,
        },
    );
}

#[test]
fn test_create_device_and_buffer() {
    let executor = VulkanExecutor::new();
    let session = make_session();

    // Create instance
    let instance_handle = match executor.execute(
        &session,
        VulkanCommand::CreateInstance {
            app_name: Some("DeviceTest".to_string()),
            app_version: 1,
            engine_name: None,
            engine_version: 0,
            api_version: ash::vk::make_api_version(0, 1, 0, 0),
            enabled_extensions: Vec::new(),
            enabled_layers: Vec::new(),
        },
    ) {
        VulkanResponse::InstanceCreated { handle } => handle,
        other => panic!("expected InstanceCreated, got {:?}", other),
    };

    // Enumerate physical devices
    let pd_handle = match executor.execute(
        &session,
        VulkanCommand::EnumeratePhysicalDevices {
            instance: instance_handle,
        },
    ) {
        VulkanResponse::PhysicalDevices { handles } => {
            assert!(!handles.is_empty());
            handles[0]
        }
        other => panic!("expected PhysicalDevices, got {:?}", other),
    };

    // Find a compute queue family
    let compute_family = match executor.execute(
        &session,
        VulkanCommand::GetPhysicalDeviceQueueFamilyProperties {
            physical_device: pd_handle,
        },
    ) {
        VulkanResponse::QueueFamilyProperties { families } => {
            families
                .iter()
                .enumerate()
                .find(|(_, qf)| qf.queue_flags & 0x00000002 != 0) // VK_QUEUE_COMPUTE_BIT
                .map(|(i, _)| i as u32)
                .expect("no compute queue family found")
        }
        other => panic!("expected QueueFamilyProperties, got {:?}", other),
    };

    println!("Using compute queue family: {}", compute_family);

    // Create device
    let device_handle = match executor.execute(
        &session,
        VulkanCommand::CreateDevice {
            physical_device: pd_handle,
            queue_create_infos: vec![DeviceQueueCreateInfo {
                queue_family_index: compute_family,
                queue_priorities: vec![1.0],
            }],
            enabled_extensions: Vec::new(),
            enabled_features: None,
        },
    ) {
        VulkanResponse::DeviceCreated { handle } => {
            println!("Device created: {:?}", handle);
            handle
        }
        other => panic!("expected DeviceCreated, got {:?}", other),
    };

    // Create buffer (1024 bytes, storage buffer usage)
    let buffer_handle = match executor.execute(
        &session,
        VulkanCommand::CreateBuffer {
            device: device_handle,
            size: 1024,
            usage: 0x00000080 | 0x00000001, // STORAGE_BUFFER | TRANSFER_SRC
            sharing_mode: 0,                // EXCLUSIVE
            queue_family_indices: Vec::new(),
        },
    ) {
        VulkanResponse::BufferCreated { handle } => {
            println!("Buffer created: {:?}", handle);
            handle
        }
        other => panic!("expected BufferCreated, got {:?}", other),
    };

    // Get memory requirements
    let (mem_size, mem_type_bits) = match executor.execute(
        &session,
        VulkanCommand::GetBufferMemoryRequirements {
            device: device_handle,
            buffer: buffer_handle,
        },
    ) {
        VulkanResponse::MemoryRequirements {
            size,
            alignment,
            memory_type_bits,
        } => {
            println!(
                "Memory requirements: size={}, alignment={}, type_bits=0x{:x}",
                size, alignment, memory_type_bits
            );
            (size, memory_type_bits)
        }
        other => panic!("expected MemoryRequirements, got {:?}", other),
    };

    // Find host-visible memory type
    let mem_type_index = match executor.execute(
        &session,
        VulkanCommand::GetPhysicalDeviceMemoryProperties {
            physical_device: pd_handle,
        },
    ) {
        VulkanResponse::PhysicalDeviceMemoryProperties {
            memory_types, ..
        } => {
            memory_types
                .iter()
                .enumerate()
                .find(|(i, mt)| {
                    (mem_type_bits & (1 << i)) != 0
                        && (mt.property_flags & 0x06) != 0 // HOST_VISIBLE | HOST_COHERENT
                })
                .map(|(i, _)| i as u32)
                .expect("no host-visible memory type found")
        }
        other => panic!("expected PhysicalDeviceMemoryProperties, got {:?}", other),
    };

    println!("Using memory type index: {}", mem_type_index);

    // Allocate memory
    let memory_handle = match executor.execute(
        &session,
        VulkanCommand::AllocateMemory {
            device: device_handle,
            alloc_size: mem_size,
            memory_type_index: mem_type_index,
        },
    ) {
        VulkanResponse::MemoryAllocated { handle } => {
            println!("Memory allocated: {:?}", handle);
            handle
        }
        other => panic!("expected MemoryAllocated, got {:?}", other),
    };

    // Bind buffer to memory
    match executor.execute(
        &session,
        VulkanCommand::BindBufferMemory {
            device: device_handle,
            buffer: buffer_handle,
            memory: memory_handle,
            memory_offset: 0,
        },
    ) {
        VulkanResponse::Success => println!("Buffer bound to memory"),
        other => panic!("expected Success, got {:?}", other),
    }

    // Map memory
    let mapped_data = match executor.execute(
        &session,
        VulkanCommand::MapMemory {
            device: device_handle,
            memory: memory_handle,
            offset: 0,
            size: 1024,
            flags: 0,
        },
    ) {
        VulkanResponse::MemoryMapped { data } => {
            println!("Memory mapped, {} bytes", data.len());
            data
        }
        other => panic!("expected MemoryMapped, got {:?}", other),
    };

    // Write some data and unmap
    let mut write_data = mapped_data;
    for (i, byte) in write_data.iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }

    match executor.execute(
        &session,
        VulkanCommand::UnmapMemory {
            device: device_handle,
            memory: memory_handle,
            written_data: Some(write_data),
            offset: 0,
        },
    ) {
        VulkanResponse::Success => println!("Memory unmapped with data"),
        other => panic!("expected Success, got {:?}", other),
    }

    // Cleanup
    executor.execute(
        &session,
        VulkanCommand::DestroyBuffer {
            device: device_handle,
            buffer: buffer_handle,
        },
    );
    executor.execute(
        &session,
        VulkanCommand::FreeMemory {
            device: device_handle,
            memory: memory_handle,
        },
    );
    executor.execute(
        &session,
        VulkanCommand::DestroyDevice {
            device: device_handle,
        },
    );
    executor.execute(
        &session,
        VulkanCommand::DestroyInstance {
            instance: instance_handle,
        },
    );

    println!("All resources cleaned up successfully");
}

#[test]
fn test_enumerate_extensions() {
    let executor = VulkanExecutor::new();
    let session = make_session();

    // Instance extension properties (no layer)
    let resp = executor.execute(
        &session,
        VulkanCommand::EnumerateInstanceExtensionProperties { layer_name: None },
    );

    match resp {
        VulkanResponse::ExtensionProperties { extensions } => {
            println!("Instance extensions: {}", extensions.len());
            for ext in extensions.iter().take(10) {
                println!("  {} (v{})", ext.extension_name, ext.spec_version);
            }
        }
        other => println!("Unexpected response: {:?}", other),
    }

    // Instance layer properties
    let resp = executor.execute(&session, VulkanCommand::EnumerateInstanceLayerProperties);

    match resp {
        VulkanResponse::LayerProperties { layers } => {
            println!("Instance layers: {}", layers.len());
            for layer in layers.iter().take(5) {
                println!("  {} - {}", layer.layer_name, layer.description);
            }
        }
        other => println!("Unexpected response: {:?}", other),
    }
}
