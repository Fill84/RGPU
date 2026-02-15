use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::watch;
use tokio::task::JoinHandle as TokioJoinHandle;
use tracing::{debug, error, info};

use rgpu_core::config::ServerConfig;
use rgpu_protocol::messages::{Message, PROTOCOL_VERSION};
use rgpu_protocol::wire;

use crate::state::{
    LocalServerStatus, MetricsSnapshot, ServerConnectionState, ServerState, UiState,
};

/// Spawns a background thread that periodically polls all configured servers
/// for GPU info and metrics. Also manages the embedded server lifecycle.
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
async fn connect_and_auth(
    address: &str,
    token: &str,
) -> anyhow::Result<(ServerConnection, Option<u16>, Vec<rgpu_protocol::gpu_info::GpuInfo>)> {
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

/// State for the embedded server running inside the fetcher.
struct EmbeddedServer {
    shutdown_tx: watch::Sender<bool>,
    task_handle: TokioJoinHandle<()>,
    /// The address of the embedded server (used to identify its ServerState entry).
    address: String,
}

/// Main fetcher loop: manages embedded server, dynamic connections, and polls periodically.
async fn fetcher_loop(state: Arc<Mutex<UiState>>, ctx: egui::Context) {
    let poll_interval = {
        let st = state.lock().unwrap();
        st.poll_interval_secs
    };

    // Per-server connections, kept in sync with state.servers
    let mut connections: Vec<Option<ServerConnection>> = {
        let st = state.lock().unwrap();
        (0..st.servers.len()).map(|_| None).collect()
    };

    let mut embedded_server: Option<EmbeddedServer> = None;

    loop {
        // Check if we should stop
        {
            let st = state.lock().unwrap();
            if st.should_stop {
                break;
            }
        }

        // --- Handle embedded server lifecycle ---
        handle_server_lifecycle(&state, &ctx, &mut embedded_server, &mut connections).await;

        // --- Handle dynamic connections ---
        handle_pending_connections(&state, &mut connections);
        handle_disconnect_requests(&state, &mut connections);

        // --- Poll all servers ---
        let server_count = {
            let st = state.lock().unwrap();
            st.servers.len()
        };

        for i in 0..server_count {
            // Get address and token for this server
            let (address, token) = {
                let st = state.lock().unwrap();
                if i >= st.servers.len() {
                    break; // servers list shrank
                }
                (st.servers[i].address.clone(), st.servers[i].token.clone())
            };

            // Try to (re)connect if needed
            if i < connections.len() && connections[i].is_none() {
                {
                    let mut st = state.lock().unwrap();
                    if i < st.servers.len() {
                        st.servers[i].connection_state = ServerConnectionState::Connecting;
                    }
                }
                ctx.request_repaint();

                match connect_and_auth(&address, &token).await {
                    Ok((conn, server_id, gpus)) => {
                        if i < connections.len() {
                            connections[i] = Some(conn);
                        }
                        let mut st = state.lock().unwrap();
                        if i < st.servers.len() {
                            st.servers[i].connection_state = ServerConnectionState::Connected;
                            st.servers[i].server_id = server_id;
                            st.servers[i].gpus = gpus;
                        }
                        debug!("connected to server {}", address);
                    }
                    Err(e) => {
                        let mut st = state.lock().unwrap();
                        if i < st.servers.len() {
                            st.servers[i].connection_state =
                                ServerConnectionState::Error(e.to_string());
                            st.push_error(format!("connect {}: {}", address, e));
                        }
                        debug!("failed to connect to {}: {}", address, e);
                    }
                }
                ctx.request_repaint();
            }

            // Poll connected servers
            if i < connections.len() {
                if let Some(conn) = &mut connections[i] {
                    // Query GPUs
                    match conn.request(&Message::QueryGpus).await {
                        Ok(Message::GpuList(gpus)) => {
                            let mut st = state.lock().unwrap();
                            if i < st.servers.len() {
                                st.servers[i].gpus = gpus;
                            }
                        }
                        Ok(_) => {}
                        Err(e) => {
                            connections[i] = None;
                            let mut st = state.lock().unwrap();
                            if i < st.servers.len() {
                                st.servers[i].connection_state =
                                    ServerConnectionState::Error(e.to_string());
                                st.push_error(format!("query gpus {}: {}", address, e));
                            }
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
                                if i < st.servers.len() {
                                    st.servers[i].server_id = Some(server_id);
                                    st.servers[i].push_metrics(snapshot);
                                }
                            }
                            Ok(_) => {
                                // Server doesn't support metrics (older version)
                            }
                            Err(e) => {
                                connections[i] = None;
                                let mut st = state.lock().unwrap();
                                if i < st.servers.len() {
                                    st.servers[i].connection_state =
                                        ServerConnectionState::Error(e.to_string());
                                    st.push_error(format!("query metrics {}: {}", address, e));
                                }
                                ctx.request_repaint();
                                continue;
                            }
                        }
                    }

                    ctx.request_repaint();
                }
            }
        }

        tokio::time::sleep(std::time::Duration::from_secs(poll_interval)).await;
    }

    // Cleanup: stop embedded server on exit
    if let Some(srv) = embedded_server.take() {
        let _ = srv.shutdown_tx.send(true);
        let _ = srv.task_handle.await;
    }
}

/// Handle embedded server start/stop requests.
async fn handle_server_lifecycle(
    state: &Arc<Mutex<UiState>>,
    ctx: &egui::Context,
    embedded_server: &mut Option<EmbeddedServer>,
    connections: &mut Vec<Option<ServerConnection>>,
) {
    let (start_requested, stop_requested) = {
        let st = state.lock().unwrap();
        (st.server_start_requested, st.server_stop_requested)
    };

    // Handle start request
    if start_requested {
        {
            let mut st = state.lock().unwrap();
            st.server_start_requested = false;
        }

        // Build server config from UI state
        let (server_config, tokens, address) = {
            let st = state.lock().unwrap();
            let cfg = &st.local_server_config;
            let server_config = ServerConfig {
                server_id: cfg.server_id,
                port: cfg.port,
                bind: cfg.bind.clone(),
                transport: cfg.transport.clone(),
                cert_path: if cfg.cert_path.is_empty() {
                    None
                } else {
                    Some(cfg.cert_path.clone())
                },
                key_path: if cfg.key_path.is_empty() {
                    None
                } else {
                    Some(cfg.key_path.clone())
                },
                expose_gpus: None,
                max_clients: cfg.max_clients,
            };
            let tokens = cfg.tokens.clone();
            let address = format!("127.0.0.1:{}", cfg.port);
            (server_config, tokens, address)
        };

        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        // Spawn the server
        let server = rgpu_server::RgpuServer::new(server_config, tokens);
        let task_handle = tokio::spawn(async move {
            if let Err(e) = server.run_with_shutdown(shutdown_rx).await {
                error!("embedded server error: {}", e);
            }
        });

        info!("embedded server starting on {}", address);

        // Add a self-monitoring connection entry
        let token = {
            let st = state.lock().unwrap();
            st.local_server_config
                .tokens
                .first()
                .map(|t| t.token.clone())
                .unwrap_or_default()
        };

        {
            let mut st = state.lock().unwrap();
            st.local_server_status = LocalServerStatus::Running;
            st.servers.push(ServerState::new(address.clone(), token));
        }
        connections.push(None);

        *embedded_server = Some(EmbeddedServer {
            shutdown_tx,
            task_handle,
            address,
        });

        // Short delay to let the server start before we try to connect
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        ctx.request_repaint();
    }

    // Handle stop request
    if stop_requested {
        {
            let mut st = state.lock().unwrap();
            st.server_stop_requested = false;
        }

        if let Some(srv) = embedded_server.take() {
            info!("stopping embedded server");
            let _ = srv.shutdown_tx.send(true);

            // Wait for the server task to complete (with timeout)
            let _ = tokio::time::timeout(
                std::time::Duration::from_secs(15),
                srv.task_handle,
            )
            .await;

            // Remove the self-monitoring connection
            let mut st = state.lock().unwrap();
            if let Some(idx) = st
                .servers
                .iter()
                .position(|s| s.address == srv.address)
            {
                st.servers.remove(idx);
                if idx < connections.len() {
                    connections.remove(idx);
                }
            }
            st.local_server_status = LocalServerStatus::Stopped;
        }
        ctx.request_repaint();
    }
}

/// Process pending connection requests from the UI.
fn handle_pending_connections(
    state: &Arc<Mutex<UiState>>,
    connections: &mut Vec<Option<ServerConnection>>,
) {
    let pending = {
        let mut st = state.lock().unwrap();
        std::mem::take(&mut st.pending_connections)
    };

    if !pending.is_empty() {
        let mut st = state.lock().unwrap();
        for pc in pending {
            st.servers
                .push(ServerState::new(pc.address, pc.token));
            connections.push(None);
        }
    }
}

/// Process disconnect requests from the UI.
fn handle_disconnect_requests(
    state: &Arc<Mutex<UiState>>,
    connections: &mut Vec<Option<ServerConnection>>,
) {
    let mut st = state.lock().unwrap();

    // Collect indices to disconnect (in reverse order to preserve indices)
    let mut to_remove: Vec<usize> = Vec::new();
    for (i, server) in st.servers.iter().enumerate() {
        if server.should_disconnect {
            to_remove.push(i);
        }
    }

    // Remove in reverse order
    for &idx in to_remove.iter().rev() {
        if idx < connections.len() {
            connections.remove(idx);
        }
        st.servers.remove(idx);
    }
}
