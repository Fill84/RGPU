use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::net::TcpListener;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

use rgpu_protocol::gpu_info::GpuInfo;
use rgpu_protocol::messages::{Message, PROTOCOL_VERSION};

use rgpu_core::config::{ServerConfig, TransportMode};
use rgpu_transport::auth;
use rgpu_transport::connection::RgpuConnection;
use rgpu_transport::tls;

use crate::cuda_executor::CudaExecutor;
use crate::vulkan_executor::VulkanExecutor;
use crate::gpu_discovery;
use crate::session::Session;

/// Server-wide metrics tracked via atomic counters.
pub struct ServerMetrics {
    pub connections_total: AtomicU64,
    pub connections_active: AtomicU32,
    pub requests_total: AtomicU64,
    pub errors_total: AtomicU64,
    pub cuda_commands: AtomicU64,
    pub vulkan_commands: AtomicU64,
    pub start_time: std::time::Instant,
    pub bind_address: parking_lot::RwLock<String>,
}

impl ServerMetrics {
    fn new() -> Self {
        Self {
            connections_total: AtomicU64::new(0),
            connections_active: AtomicU32::new(0),
            requests_total: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            cuda_commands: AtomicU64::new(0),
            vulkan_commands: AtomicU64::new(0),
            start_time: std::time::Instant::now(),
            bind_address: parking_lot::RwLock::new(String::new()),
        }
    }
}

/// The main RGPU server. Listens for incoming connections and serves GPU commands.
pub struct RgpuServer {
    config: ServerConfig,
    gpu_infos: Vec<GpuInfo>,
    cuda_executor: Arc<CudaExecutor>,
    vulkan_executor: Arc<VulkanExecutor>,
    next_session_id: AtomicU32,
    /// Accepted authentication tokens (empty = no auth required)
    accepted_tokens: Vec<rgpu_core::config::TokenEntry>,
    metrics: Arc<ServerMetrics>,
}

impl RgpuServer {
    /// Create a new server, discovering local GPUs.
    pub fn new(
        config: ServerConfig,
        accepted_tokens: Vec<rgpu_core::config::TokenEntry>,
    ) -> Self {
        let gpu_infos = gpu_discovery::discover_gpus(config.server_id);
        let cuda_executor = Arc::new(CudaExecutor::new(gpu_infos.clone()));
        let vulkan_executor = Arc::new(VulkanExecutor::new());

        Self {
            config,
            gpu_infos,
            cuda_executor,
            vulkan_executor,
            next_session_id: AtomicU32::new(1),
            accepted_tokens,
            metrics: Arc::new(ServerMetrics::new()),
        }
    }

    /// Start listening for connections.
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!(
            "serving {} GPU(s): {}",
            self.gpu_infos.len(),
            self.gpu_infos
                .iter()
                .map(|g| g.device_name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );

        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        tokio::spawn(async move {
            shutdown_signal().await;
            let _ = shutdown_tx.send(true);
        });

        // Spawn periodic metrics logger
        let metrics = self.metrics.clone();
        let mut metrics_shutdown = shutdown_rx.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(60)) => {
                        info!(
                            connections_total = metrics.connections_total.load(Ordering::Relaxed),
                            connections_active = metrics.connections_active.load(Ordering::Relaxed),
                            requests = metrics.requests_total.load(Ordering::Relaxed),
                            errors = metrics.errors_total.load(Ordering::Relaxed),
                            cuda = metrics.cuda_commands.load(Ordering::Relaxed),
                            vulkan = metrics.vulkan_commands.load(Ordering::Relaxed),
                            "metrics snapshot"
                        );
                    }
                    _ = metrics_shutdown.changed() => { break; }
                }
            }
        });

        // Store bind address for metrics queries
        *self.metrics.bind_address.write() =
            format!("{}:{}", self.config.bind, self.config.port);

        match self.config.transport {
            TransportMode::Quic => self.run_quic(shutdown_rx).await,
            TransportMode::Tcp => self.run_tcp(shutdown_rx).await,
        }
    }

    /// Run with TCP transport (plain or TLS).
    async fn run_tcp(
        &self,
        mut shutdown_rx: watch::Receiver<bool>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let bind_addr = format!("{}:{}", self.config.bind, self.config.port);
        let listener = TcpListener::bind(&bind_addr).await?;

        info!("RGPU server listening on {} (TCP)", bind_addr);

        // Build TLS acceptor if cert/key are provided
        let tls_acceptor = if let (Some(cert), Some(key)) =
            (&self.config.cert_path, &self.config.key_path)
        {
            Some(tls::build_server_tls(cert, key)?)
        } else {
            warn!("no TLS certificate configured - connections will be unencrypted");
            None
        };

        let active_sessions = Arc::new(AtomicU32::new(0));
        let max_clients = self.config.max_clients;

        loop {
            let tcp_accept = listener.accept();
            let shutdown = shutdown_rx.changed();

            tokio::select! {
                result = tcp_accept => {
                    let (tcp_stream, peer_addr) = result?;
                    info!("new connection from {}", peer_addr);

                    // Enforce connection limit
                    let current = active_sessions.load(Ordering::Relaxed);
                    if current >= max_clients {
                        warn!("connection from {} rejected: max_clients ({}) reached", peer_addr, max_clients);
                        drop(tcp_stream);
                        continue;
                    }

                    let cuda_executor = self.cuda_executor.clone();
                    let vulkan_executor = self.vulkan_executor.clone();
                    let gpu_infos = self.gpu_infos.clone();
                    let session_id = self.next_session_id.fetch_add(1, Ordering::Relaxed);
                    let server_id = self.config.server_id;
                    let accepted_tokens = self.accepted_tokens.clone();
                    let active = active_sessions.clone();
                    let metrics = self.metrics.clone();

                    active.fetch_add(1, Ordering::Relaxed);
                    metrics.connections_total.fetch_add(1, Ordering::Relaxed);
                    metrics.connections_active.fetch_add(1, Ordering::Relaxed);

                    if let Some(ref acceptor) = tls_acceptor {
                        let acceptor = acceptor.clone();
                        tokio::spawn(async move {
                            match acceptor.accept(tcp_stream).await {
                                Ok(tls_stream) => {
                                    match RgpuConnection::from_server_stream(tls_stream).await {
                                        Ok(conn) => {
                                            Self::handle_client(
                                                conn,
                                                session_id,
                                                server_id,
                                                gpu_infos,
                                                cuda_executor,
                                                vulkan_executor,
                                                accepted_tokens,
                                                metrics.clone(),
                                            )
                                            .await;
                                        }
                                        Err(e) => error!("connection setup failed: {}", e),
                                    }
                                }
                                Err(e) => error!("TLS handshake failed from {}: {}", peer_addr, e),
                            }
                            active.fetch_sub(1, Ordering::Relaxed);
                            metrics.connections_active.fetch_sub(1, Ordering::Relaxed);
                        });
                    } else {
                        // No TLS - for development/testing only
                        let gpu_infos_clone = gpu_infos;
                        tokio::spawn(async move {
                            Self::handle_plain_client(
                                tcp_stream,
                                session_id,
                                server_id,
                                gpu_infos_clone,
                                cuda_executor,
                                vulkan_executor,
                                accepted_tokens,
                                metrics.clone(),
                            )
                            .await;
                            active.fetch_sub(1, Ordering::Relaxed);
                            metrics.connections_active.fetch_sub(1, Ordering::Relaxed);
                        });
                    }
                }
                _ = shutdown => {
                    info!("shutdown signal received, stopping TCP accept loop");
                    break;
                }
            }
        }

        // Wait for active sessions to drain (max 10s)
        let remaining = active_sessions.load(Ordering::Relaxed);
        if remaining > 0 {
            info!("waiting for {} active session(s) to finish (max 10s)", remaining);
            let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
            while active_sessions.load(Ordering::Relaxed) > 0
                && tokio::time::Instant::now() < deadline
            {
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            let still_active = active_sessions.load(Ordering::Relaxed);
            if still_active > 0 {
                warn!("{} session(s) still active after drain timeout", still_active);
            }
        }

        info!("server shut down cleanly");
        Ok(())
    }

    /// Run with QUIC transport (always encrypted).
    async fn run_quic(
        &self,
        mut shutdown_rx: watch::Receiver<bool>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (cert_path, key_path) = match (&self.config.cert_path, &self.config.key_path) {
            (Some(c), Some(k)) => (c.as_str(), k.as_str()),
            _ => return Err("QUIC transport requires cert_path and key_path".into()),
        };

        let bind_addr: std::net::SocketAddr =
            format!("{}:{}", self.config.bind, self.config.port).parse()?;

        let endpoint = rgpu_transport::quic::build_quic_server(bind_addr, cert_path, key_path)?;
        info!("RGPU server listening on {} (QUIC)", bind_addr);

        let active_sessions = Arc::new(AtomicU32::new(0));
        let max_clients = self.config.max_clients;

        loop {
            tokio::select! {
                incoming = endpoint.accept() => {
                    let incoming = match incoming {
                        Some(i) => i,
                        None => break,
                    };

                    // Enforce connection limit
                    let current = active_sessions.load(Ordering::Relaxed);
                    if current >= max_clients {
                        warn!("QUIC connection rejected: max_clients ({}) reached", max_clients);
                        incoming.refuse();
                        continue;
                    }

                    let cuda_executor = self.cuda_executor.clone();
                    let vulkan_executor = self.vulkan_executor.clone();
                    let gpu_infos = self.gpu_infos.clone();
                    let session_id = self.next_session_id.fetch_add(1, Ordering::Relaxed);
                    let server_id = self.config.server_id;
                    let accepted_tokens = self.accepted_tokens.clone();
                    let active = active_sessions.clone();
                    let metrics = self.metrics.clone();

                    active.fetch_add(1, Ordering::Relaxed);
                    metrics.connections_total.fetch_add(1, Ordering::Relaxed);
                    metrics.connections_active.fetch_add(1, Ordering::Relaxed);

                    tokio::spawn(async move {
                        match incoming.await {
                            Ok(connection) => {
                                let remote = connection.remote_address();
                                info!(session_id, "QUIC client connected from {}", remote);

                                Self::handle_quic_client(
                                    connection,
                                    session_id,
                                    server_id,
                                    gpu_infos,
                                    cuda_executor,
                                    vulkan_executor,
                                    accepted_tokens,
                                    metrics.clone(),
                                )
                                .await;

                                info!(session_id, "QUIC client {} disconnected", remote);
                            }
                            Err(e) => {
                                error!("QUIC incoming connection error: {}", e);
                            }
                        }
                        active.fetch_sub(1, Ordering::Relaxed);
                        metrics.connections_active.fetch_sub(1, Ordering::Relaxed);
                    });
                }
                _ = shutdown_rx.changed() => {
                    info!("shutdown signal received, stopping QUIC accept loop");
                    break;
                }
            }
        }

        // Close the endpoint to reject new connections
        endpoint.close(quinn::VarInt::from_u32(0), b"server shutting down");

        // Wait for active sessions to drain (max 10s)
        let remaining = active_sessions.load(Ordering::Relaxed);
        if remaining > 0 {
            info!("waiting for {} active session(s) to finish (max 10s)", remaining);
            let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
            while active_sessions.load(Ordering::Relaxed) > 0
                && tokio::time::Instant::now() < deadline
            {
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            let still_active = active_sessions.load(Ordering::Relaxed);
            if still_active > 0 {
                warn!("{} session(s) still active after drain timeout", still_active);
            }
        }

        info!("server shut down cleanly");
        Ok(())
    }

    /// Handle a plain TCP client (no TLS) - development only.
    async fn handle_plain_client(
        stream: tokio::net::TcpStream,
        session_id: u32,
        server_id: u16,
        gpu_infos: Vec<GpuInfo>,
        cuda_executor: Arc<CudaExecutor>,
        vulkan_executor: Arc<VulkanExecutor>,
        _accepted_tokens: Vec<rgpu_core::config::TokenEntry>,
        metrics: Arc<ServerMetrics>,
    ) {
        use rgpu_protocol::wire;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let session = Session::new(session_id, server_id, "unknown".to_string());
        let (mut reader, mut writer) = stream.into_split();

        info!(session_id, "plain TCP client connected");

        let mut header_buf = [0u8; rgpu_protocol::wire::HEADER_SIZE];

        loop {
            // Read frame header with idle timeout
            match tokio::time::timeout(
                Duration::from_secs(120),
                reader.read_exact(&mut header_buf),
            )
            .await
            {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => {
                    info!(session_id, "client disconnected: {}", e);
                    break;
                }
                Err(_) => {
                    warn!(session_id, "client idle timeout (120s), disconnecting");
                    break;
                }
            }

            let (flags, _stream_id, payload_len) = match wire::decode_header(&header_buf) {
                Ok(v) => v,
                Err(e) => {
                    error!(session_id, "invalid frame: {}", e);
                    break;
                }
            };

            // Read payload
            let mut payload = vec![0u8; payload_len as usize];
            if let Err(e) = reader.read_exact(&mut payload).await {
                error!(session_id, "payload read error: {}", e);
                break;
            }

            // Decode and handle message
            let msg = match wire::decode_message(&payload, flags) {
                Ok(m) => m,
                Err(e) => {
                    error!(session_id, "decode error: {}", e);
                    continue;
                }
            };

            let response = Self::handle_message(&session, msg, &gpu_infos, &cuda_executor, &vulkan_executor, &metrics);

            // Send response
            if let Some(resp) = response {
                match wire::encode_message(&resp, 0) {
                    Ok(frame) => {
                        if let Err(e) = writer.write_all(&frame).await {
                            error!(session_id, "write error: {}", e);
                            break;
                        }
                    }
                    Err(e) => {
                        error!(session_id, "encode error: {}", e);
                    }
                }
            }
        }

        // Clean up leaked resources
        let leaked = session.all_handles().len();
        if leaked > 0 {
            warn!(session_id, "{} handle(s) leaked at disconnect, cleaning up", leaked);
        }
        cuda_executor.cleanup_session(&session);
        vulkan_executor.cleanup_session(&session);
        info!(session_id, "client session ended");
    }

    /// Handle a TLS client connection.
    async fn handle_client(
        conn: RgpuConnection,
        session_id: u32,
        server_id: u16,
        gpu_infos: Vec<GpuInfo>,
        cuda_executor: Arc<CudaExecutor>,
        vulkan_executor: Arc<VulkanExecutor>,
        _accepted_tokens: Vec<rgpu_core::config::TokenEntry>,
        metrics: Arc<ServerMetrics>,
    ) {
        let session = Session::new(session_id, server_id, "tls-client".to_string());
        info!(session_id, "TLS client connected");

        loop {
            match tokio::time::timeout(Duration::from_secs(120), conn.recv()).await {
                Ok(Ok(msg)) => {
                    let response =
                        Self::handle_message(&session, msg, &gpu_infos, &cuda_executor, &vulkan_executor, &metrics);
                    if let Some(resp) = response {
                        if let Err(e) = conn.send(resp).await {
                            error!(session_id, "send error: {}", e);
                            break;
                        }
                    }
                }
                Ok(Err(e)) => {
                    info!(session_id, "client disconnected: {}", e);
                    break;
                }
                Err(_) => {
                    warn!(session_id, "client idle timeout (120s), disconnecting");
                    break;
                }
            }
        }

        let leaked = session.all_handles().len();
        if leaked > 0 {
            warn!(session_id, "{} handle(s) leaked at disconnect, cleaning up", leaked);
        }
        cuda_executor.cleanup_session(&session);
        vulkan_executor.cleanup_session(&session);
        info!(session_id, "client session ended");
    }

    /// Handle a QUIC client connection.
    /// Each bidirectional stream carries one request-response pair.
    async fn handle_quic_client(
        connection: quinn::Connection,
        session_id: u32,
        server_id: u16,
        gpu_infos: Vec<GpuInfo>,
        cuda_executor: Arc<CudaExecutor>,
        vulkan_executor: Arc<VulkanExecutor>,
        _accepted_tokens: Vec<rgpu_core::config::TokenEntry>,
        metrics: Arc<ServerMetrics>,
    ) {
        use rgpu_protocol::wire;

        let session = Arc::new(Session::new(session_id, server_id, "quic-client".to_string()));

        loop {
            match connection.accept_bi().await {
                Ok((mut send, mut recv)) => {
                    let cuda_exec = cuda_executor.clone();
                    let vulkan_exec = vulkan_executor.clone();
                    let gpu_infos = gpu_infos.clone();
                    let session = session.clone();
                    let metrics = metrics.clone();

                    tokio::spawn(async move {
                        // Read request
                        let mut header_buf = [0u8; wire::HEADER_SIZE];
                        if let Err(e) = recv.read_exact(&mut header_buf).await {
                            debug!("QUIC stream read header error: {}", e);
                            return;
                        }

                        let (flags, _, payload_len) = match wire::decode_header(&header_buf) {
                            Ok(v) => v,
                            Err(e) => {
                                error!(session_id, "QUIC invalid frame: {}", e);
                                return;
                            }
                        };

                        let mut payload = vec![0u8; payload_len as usize];
                        if let Err(e) = recv.read_exact(&mut payload).await {
                            error!(session_id, "QUIC payload read error: {}", e);
                            return;
                        }

                        let msg = match wire::decode_message(&payload, flags) {
                            Ok(m) => m,
                            Err(e) => {
                                error!(session_id, "QUIC decode error: {}", e);
                                return;
                            }
                        };

                        // Handle and respond
                        if let Some(resp) = Self::handle_message(
                            &session, msg, &gpu_infos, &cuda_exec, &vulkan_exec, &metrics,
                        ) {
                            match wire::encode_message(&resp, 0) {
                                Ok(frame) => {
                                    if let Err(e) = send.write_all(&frame).await {
                                        debug!("QUIC write error: {}", e);
                                    }
                                    let _ = send.finish();
                                }
                                Err(e) => {
                                    error!(session_id, "QUIC encode error: {}", e);
                                }
                            }
                        }
                    });
                }
                Err(quinn::ConnectionError::ApplicationClosed(_)) => break,
                Err(e) => {
                    debug!(session_id, "QUIC accept stream error: {}", e);
                    break;
                }
            }
        }

        // Clean up leaked resources
        let leaked = session.all_handles().len();
        if leaked > 0 {
            warn!(session_id, "{} handle(s) leaked at disconnect, cleaning up", leaked);
        }
        cuda_executor.cleanup_session(&session);
        vulkan_executor.cleanup_session(&session);
        info!(session_id, "QUIC client session ended");
    }

    /// Process a single message and return the response.
    fn handle_message(
        session: &Session,
        msg: Message,
        gpu_infos: &[GpuInfo],
        cuda_executor: &CudaExecutor,
        vulkan_executor: &VulkanExecutor,
        metrics: &ServerMetrics,
    ) -> Option<Message> {
        metrics.requests_total.fetch_add(1, Ordering::Relaxed);

        match &msg {
            Message::CudaCommand { .. } | Message::CudaBatch(_) => {
                metrics.cuda_commands.fetch_add(1, Ordering::Relaxed);
            }
            Message::VulkanCommand { .. } => {
                metrics.vulkan_commands.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        match msg {
            Message::Hello {
                protocol_version,
                name,
                ..
            } => {
                info!(
                    session_id = session.session_id,
                    "Hello from '{}' (protocol v{})", name, protocol_version
                );
                let challenge = auth::generate_challenge(32);
                Some(Message::Hello {
                    protocol_version: PROTOCOL_VERSION,
                    name: "RGPU Server".to_string(),
                    challenge: Some(challenge),
                })
            }

            Message::Authenticate { .. } => {
                // For now, accept any auth in Phase 1
                info!(
                    session_id = session.session_id,
                    "client authenticated"
                );
                Some(Message::AuthResult {
                    success: true,
                    session_id: Some(session.session_id),
                    server_id: Some(session.server_id()),
                    available_gpus: gpu_infos.to_vec(),
                    error_message: None,
                })
            }

            Message::QueryGpus => Some(Message::GpuList(gpu_infos.to_vec())),

            Message::QueryMetrics => Some(Message::MetricsData {
                connections_total: metrics.connections_total.load(Ordering::Relaxed),
                connections_active: metrics.connections_active.load(Ordering::Relaxed),
                requests_total: metrics.requests_total.load(Ordering::Relaxed),
                errors_total: metrics.errors_total.load(Ordering::Relaxed),
                cuda_commands: metrics.cuda_commands.load(Ordering::Relaxed),
                vulkan_commands: metrics.vulkan_commands.load(Ordering::Relaxed),
                uptime_secs: metrics.start_time.elapsed().as_secs(),
                server_id: session.server_id(),
                server_address: metrics.bind_address.read().clone(),
            }),

            Message::CudaCommand {
                request_id,
                command,
            } => {
                let response = cuda_executor.execute(session, command);
                Some(Message::CudaResponse {
                    request_id,
                    response,
                })
            }

            Message::VulkanCommand {
                request_id,
                command,
            } => {
                let response = vulkan_executor.execute(session, command);
                Some(Message::VulkanResponse {
                    request_id,
                    response,
                })
            }

            Message::CudaBatch(commands) => {
                let mut last_error = None;
                for cmd in commands {
                    let response = cuda_executor.execute(session, cmd);
                    if let rgpu_protocol::cuda_commands::CudaResponse::Error { .. } = &response {
                        last_error = Some(Message::CudaResponse {
                            request_id: rgpu_protocol::messages::RequestId(0),
                            response,
                        });
                    }
                }
                // Return last error if any, otherwise Success
                Some(last_error.unwrap_or_else(|| Message::CudaResponse {
                    request_id: rgpu_protocol::messages::RequestId(0),
                    response: rgpu_protocol::cuda_commands::CudaResponse::Success,
                }))
            }

            Message::Ping => Some(Message::Pong),

            _ => {
                warn!(
                    session_id = session.session_id,
                    "unhandled message type"
                );
                None
            }
        }
    }
}

/// Wait for a shutdown signal (Ctrl+C or SIGTERM on Unix).
async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();

    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate(),
        )
        .expect("failed to install SIGTERM handler");

        tokio::select! {
            _ = ctrl_c => { info!("received Ctrl+C, initiating shutdown"); }
            _ = sigterm.recv() => { info!("received SIGTERM, initiating shutdown"); }
        }
    }

    #[cfg(not(unix))]
    {
        ctrl_c.await.expect("failed to listen for Ctrl+C");
        info!("received Ctrl+C, initiating shutdown");
    }
}
