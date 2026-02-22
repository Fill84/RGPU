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
pub async fn start_ipc_listener(
    pipe_name: &str,
    message_handler: impl Fn(Message) -> Option<Message> + Send + Sync + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use tokio::net::windows::named_pipe::{ServerOptions, PipeMode};

    info!("IPC listening on {}", pipe_name);

    let handler = std::sync::Arc::new(message_handler);

    loop {
        let server = ServerOptions::new()
            .pipe_mode(PipeMode::Byte)
            .create(pipe_name)?;

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
