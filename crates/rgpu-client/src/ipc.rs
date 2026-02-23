use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, error, info};

use rgpu_protocol::messages::Message;
use rgpu_protocol::wire;

/// IPC server that listens for connections from the Vulkan ICD and CUDA
/// interposition library. Uses named pipes on Windows and Unix domain
/// sockets on Linux/macOS.

#[cfg(unix)]
pub async fn start_ipc_listener(
    path: &str,
    message_handler: impl Fn(Message) -> Option<Message> + Send + Sync + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use tokio::net::UnixListener;

    // Remove stale socket if it exists
    let _ = std::fs::remove_file(path);

    let listener = UnixListener::bind(path)?;
    info!("IPC listening on {}", path);

    let handler = std::sync::Arc::new(message_handler);

    loop {
        let (stream, _) = listener.accept().await?;
        let handler = handler.clone();

        tokio::spawn(async move {
            let (mut reader, mut writer) = stream.into_split();
            let mut header_buf = [0u8; wire::HEADER_SIZE];

            loop {
                match reader.read_exact(&mut header_buf).await {
                    Ok(_) => {}
                    Err(_) => break,
                }

                let (flags, _stream_id, payload_len) = match wire::decode_header(&header_buf) {
                    Ok(v) => v,
                    Err(e) => {
                        error!("IPC decode error: {}", e);
                        break;
                    }
                };

                let mut payload = vec![0u8; payload_len as usize];
                if reader.read_exact(&mut payload).await.is_err() {
                    break;
                }

                let msg = match wire::decode_message(&payload, flags) {
                    Ok(m) => m,
                    Err(e) => {
                        error!("IPC message decode error: {}", e);
                        continue;
                    }
                };

                let response = match handler(msg) {
                    Some(resp) => resp,
                    None => {
                        // Fallback: send an error response so the app doesn't hang
                        error!("IPC handler returned None, sending error response");
                        Message::CudaResponse {
                            request_id: rgpu_protocol::messages::RequestId(0),
                            response: rgpu_protocol::cuda_commands::CudaResponse::Error {
                                code: 999,
                                message: "internal daemon error".to_string(),
                            },
                        }
                    }
                };

                match wire::encode_message(&response, 0) {
                    Ok(frame) => {
                        if writer.write_all(&frame).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        error!("IPC encode error: {}", e);
                    }
                }
            }

            debug!("IPC client disconnected");
        });
    }
}

#[cfg(windows)]
/// Create a named pipe instance with a null DACL security descriptor.
/// This allows ALL users (including non-admin) to connect, which is required
/// when the daemon runs as SYSTEM via a Windows service.
fn create_pipe_with_open_access(
    pipe_name: &str,
) -> Result<tokio::net::windows::named_pipe::NamedPipeServer, Box<dyn std::error::Error + Send + Sync>>
{
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::Security::{
        InitializeSecurityDescriptor, SetSecurityDescriptorDacl, SECURITY_ATTRIBUTES,
    };
    use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OVERLAPPED;
    use windows_sys::Win32::System::Pipes::{
        CreateNamedPipeW, PIPE_TYPE_BYTE, PIPE_UNLIMITED_INSTANCES, PIPE_WAIT,
    };

    // Constants not always exported by windows-sys
    const SECURITY_DESCRIPTOR_REVISION: u32 = 1;
    const PIPE_ACCESS_DUPLEX: u32 = 0x00000003;

    // Build a wide string for the pipe name
    let wide_name: Vec<u16> = pipe_name.encode_utf16().chain(std::iter::once(0)).collect();

    // Allocate a security descriptor buffer (opaque struct, PSECURITY_DESCRIPTOR = *mut c_void)
    // SECURITY_DESCRIPTOR is 40 bytes on x64, use a generous buffer
    let mut sd_buffer = [0u8; 64];
    let sd_ptr = sd_buffer.as_mut_ptr() as *mut std::ffi::c_void;

    // Initialize with a null DACL (all access allowed — appropriate for local IPC)
    let ok = unsafe { InitializeSecurityDescriptor(sd_ptr, SECURITY_DESCRIPTOR_REVISION) };
    if ok == 0 {
        return Err("InitializeSecurityDescriptor failed".into());
    }
    let ok = unsafe { SetSecurityDescriptorDacl(sd_ptr, 1, std::ptr::null_mut(), 0) };
    if ok == 0 {
        return Err("SetSecurityDescriptorDacl failed".into());
    }

    let mut sa = SECURITY_ATTRIBUTES {
        nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: sd_ptr,
        bInheritHandle: 0,
    };

    let handle = unsafe {
        CreateNamedPipeW(
            wide_name.as_ptr(),
            PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
            PIPE_TYPE_BYTE | PIPE_WAIT,
            PIPE_UNLIMITED_INSTANCES,
            65536, // out buffer size
            65536, // in buffer size
            0,     // default timeout
            &mut sa,
        )
    };

    if handle == INVALID_HANDLE_VALUE {
        return Err(format!(
            "CreateNamedPipeW failed: {}",
            std::io::Error::last_os_error()
        )
        .into());
    }

    // Safety: handle is a valid named pipe handle from CreateNamedPipeW
    let server = unsafe {
        tokio::net::windows::named_pipe::NamedPipeServer::from_raw_handle(handle as _)?
    };
    Ok(server)
}

#[cfg(windows)]
pub async fn start_ipc_listener(
    pipe_name: &str,
    message_handler: impl Fn(Message) -> Option<Message> + Send + Sync + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("IPC listening on {}", pipe_name);

    let handler = std::sync::Arc::new(message_handler);

    loop {
        let server = create_pipe_with_open_access(pipe_name)?;

        server.connect().await?;
        let handler = handler.clone();

        tokio::spawn(async move {
            let (mut reader, mut writer) = tokio::io::split(server);
            let mut header_buf = [0u8; wire::HEADER_SIZE];

            loop {
                match AsyncReadExt::read_exact(&mut reader, &mut header_buf).await {
                    Ok(_) => {}
                    Err(_) => break,
                }

                let (flags, _stream_id, payload_len) = match wire::decode_header(&header_buf) {
                    Ok(v) => v,
                    Err(e) => {
                        error!("IPC decode error: {}", e);
                        break;
                    }
                };

                let mut payload = vec![0u8; payload_len as usize];
                if AsyncReadExt::read_exact(&mut reader, &mut payload)
                    .await
                    .is_err()
                {
                    break;
                }

                let msg = match wire::decode_message(&payload, flags) {
                    Ok(m) => m,
                    Err(e) => {
                        error!("IPC message decode error: {}", e);
                        continue;
                    }
                };

                if let Some(response) = handler(msg) {
                    match wire::encode_message(&response, 0) {
                        Ok(frame) => {
                            if AsyncWriteExt::write_all(&mut writer, &frame).await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            error!("IPC encode error: {}", e);
                        }
                    }
                }
            }

            debug!("IPC client disconnected");
        });
    }
}
