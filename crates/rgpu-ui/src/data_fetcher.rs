use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::debug;

use rgpu_protocol::messages::{Message, PROTOCOL_VERSION};
use rgpu_protocol::wire;

use crate::state::{MetricsSnapshot, ServerConnectionState, UiState};

/// Spawns a background thread that periodically polls all configured servers
/// for GPU info and metrics. Updates are written to the shared `UiState`.
pub fn start_data_fetcher(
    state: Arc<Mutex<UiState>>,
    ctx: egui::Context,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("rgpu-ui-fetcher".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("failed to build tokio runtime for data fetcher");

            rt.block_on(async move {
                fetcher_loop(state, ctx).await;
            });
        })
        .expect("failed to spawn data fetcher thread")
}

/// Holds a live TCP connection to a single server.
struct ServerConnection {
    reader: tokio::io::ReadHalf<TcpStream>,
    writer: tokio::io::WriteHalf<TcpStream>,
}

impl ServerConnection {
    /// Send a message and read the response.
    async fn request(&mut self, msg: &Message) -> anyhow::Result<Message> {
        let frame = wire::encode_message(msg, 0)?;
        self.writer.write_all(&frame).await?;

        let mut header_buf = [0u8; wire::HEADER_SIZE];
        self.reader.read_exact(&mut header_buf).await?;
        let (flags, _, payload_len) = wire::decode_header(&header_buf)?;
        let mut payload = vec![0u8; payload_len as usize];
        self.reader.read_exact(&mut payload).await?;
        let response = wire::decode_message(&payload, flags)?;
        Ok(response)
    }
}

/// Connect to a server and perform Hello/Auth handshake.
async fn connect_and_auth(address: &str, token: &str) -> anyhow::Result<(ServerConnection, Option<u16>, Vec<rgpu_protocol::gpu_info::GpuInfo>)> {
    let stream = TcpStream::connect(address).await?;
    let (mut reader, mut writer) = tokio::io::split(stream);

    // Send Hello
    let hello = Message::Hello {
        protocol_version: PROTOCOL_VERSION,
        name: "RGPU UI".to_string(),
        challenge: None,
    };
    let frame = wire::encode_message(&hello, 0)?;
    writer.write_all(&frame).await?;

    // Read server Hello with challenge
    let mut header_buf = [0u8; wire::HEADER_SIZE];
    reader.read_exact(&mut header_buf).await?;
    let (flags, _, payload_len) = wire::decode_header(&header_buf)?;
    let mut payload = vec![0u8; payload_len as usize];
    reader.read_exact(&mut payload).await?;
    let server_hello = wire::decode_message(&payload, flags)?;

    let challenge = match &server_hello {
        Message::Hello { challenge, .. } => challenge.clone().unwrap_or_default(),
        _ => Vec::new(),
    };

    // Send Authenticate
    let challenge_response =
        rgpu_transport::auth::compute_challenge_response(token, &challenge);
    let auth_msg = Message::Authenticate {
        token: token.to_string(),
        challenge_response,
    };
    let frame = wire::encode_message(&auth_msg, 0)?;
    writer.write_all(&frame).await?;

    // Read AuthResult
    reader.read_exact(&mut header_buf).await?;
    let (flags, _, payload_len) = wire::decode_header(&header_buf)?;
    let mut payload = vec![0u8; payload_len as usize];
    reader.read_exact(&mut payload).await?;
    let auth_result = wire::decode_message(&payload, flags)?;

    match auth_result {
        Message::AuthResult {
            success: true,
            server_id,
            available_gpus,
            ..
        } => {
            let conn = ServerConnection { reader, writer };
            Ok((conn, server_id, available_gpus))
        }
        Message::AuthResult {
            success: false,
            error_message,
            ..
        } => {
            anyhow::bail!(
                "authentication failed: {}",
                error_message.unwrap_or_default()
            );
        }
        _ => anyhow::bail!("unexpected response during authentication"),
    }
}

/// Main fetcher loop: connects to servers and polls periodically.
async fn fetcher_loop(state: Arc<Mutex<UiState>>, ctx: egui::Context) {
    // Get server info from state
    let (server_configs, poll_interval): (Vec<(String, String)>, u64) = {
        let st = state.lock().unwrap();
        let configs = st
            .servers
            .iter()
            .map(|s| (s.address.clone(), s.token.clone()))
            .collect();
        (configs, st.poll_interval_secs)
    };

    // Maintain per-server connections
    let mut connections: Vec<Option<ServerConnection>> =
        (0..server_configs.len()).map(|_| None).collect();

    loop {
        // Check if we should stop
        {
            let st = state.lock().unwrap();
            if st.should_stop {
                break;
            }
        }

        for (i, (address, token)) in server_configs.iter().enumerate() {
            // Try to (re)connect if needed
            if connections[i].is_none() {
                {
                    let mut st = state.lock().unwrap();
                    st.servers[i].connection_state = ServerConnectionState::Connecting;
                }
                ctx.request_repaint();

                match connect_and_auth(address, token).await {
                    Ok((conn, server_id, gpus)) => {
                        connections[i] = Some(conn);
                        let mut st = state.lock().unwrap();
                        st.servers[i].connection_state = ServerConnectionState::Connected;
                        st.servers[i].server_id = server_id;
                        st.servers[i].gpus = gpus;
                        debug!("connected to server {}", address);
                    }
                    Err(e) => {
                        let mut st = state.lock().unwrap();
                        st.servers[i].connection_state =
                            ServerConnectionState::Error(e.to_string());
                        st.push_error(format!("connect {}: {}", address, e));
                        debug!("failed to connect to {}: {}", address, e);
                    }
                }
                ctx.request_repaint();
            }

            // Poll connected servers
            if let Some(conn) = &mut connections[i] {
                // Query GPUs
                match conn.request(&Message::QueryGpus).await {
                    Ok(Message::GpuList(gpus)) => {
                        let mut st = state.lock().unwrap();
                        st.servers[i].gpus = gpus;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        connections[i] = None;
                        let mut st = state.lock().unwrap();
                        st.servers[i].connection_state =
                            ServerConnectionState::Error(e.to_string());
                        st.push_error(format!("query gpus {}: {}", address, e));
                        ctx.request_repaint();
                        continue;
                    }
                }

                // Query Metrics
                if let Some(conn) = &mut connections[i] {
                    match conn.request(&Message::QueryMetrics).await {
                        Ok(Message::MetricsData {
                            connections_total,
                            connections_active,
                            requests_total,
                            errors_total,
                            cuda_commands,
                            vulkan_commands,
                            uptime_secs,
                            server_id,
                            ..
                        }) => {
                            let snapshot = MetricsSnapshot {
                                timestamp: std::time::Instant::now(),
                                connections_total,
                                connections_active,
                                requests_total,
                                errors_total,
                                cuda_commands,
                                vulkan_commands,
                                uptime_secs,
                            };
                            let mut st = state.lock().unwrap();
                            st.servers[i].server_id = Some(server_id);
                            st.servers[i].push_metrics(snapshot);
                        }
                        Ok(_) => {
                            // Server doesn't support metrics (older version)
                        }
                        Err(e) => {
                            connections[i] = None;
                            let mut st = state.lock().unwrap();
                            st.servers[i].connection_state =
                                ServerConnectionState::Error(e.to_string());
                            st.push_error(format!("query metrics {}: {}", address, e));
                            ctx.request_repaint();
                            continue;
                        }
                    }
                }

                ctx.request_repaint();
            }
        }

        tokio::time::sleep(std::time::Duration::from_secs(poll_interval)).await;
    }
}
