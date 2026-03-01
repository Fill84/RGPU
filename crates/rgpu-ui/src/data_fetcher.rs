use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use tokio::sync::watch;
use tokio::task::JoinHandle as TokioJoinHandle;
use tracing::{debug, error, info, warn};

use rgpu_core::config::ServerConfig;

use crate::service_control;
use crate::service_detection;
use crate::state::{
    ClientRemoteServer, MetricsSnapshot, ServiceOrigin, ServiceStatus, UiState,
};

/// Spawns a background thread that periodically probes server and client daemon,
/// manages embedded server lifecycle, and updates shared UI state.
pub fn start_data_fetcher(state: Arc<Mutex<UiState>>, ctx: egui::Context) -> JoinHandle<()> {
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

// ============================================================================
// Embedded server state
// ============================================================================

struct EmbeddedServer {
    shutdown_tx: watch::Sender<bool>,
    task_handle: TokioJoinHandle<()>,
    server_ref: Arc<rgpu_server::RgpuServer>,
}

// ============================================================================
// Main fetcher loop
// ============================================================================

async fn fetcher_loop(state: Arc<Mutex<UiState>>, ctx: egui::Context) {
    let poll_interval = {
        let st = state.lock().unwrap();
        st.poll_interval_secs
    };

    let mut embedded_server: Option<EmbeddedServer> = None;
    let mut interpose_check_counter: u64 = 0;

    loop {
        // Check if we should stop
        {
            let st = state.lock().unwrap();
            if st.should_stop {
                break;
            }
        }

        // 1. Handle server lifecycle (start/stop embedded or service)
        handle_server_lifecycle(&state, &ctx, &mut embedded_server).await;

        // 2. Handle client lifecycle (start/stop service or process)
        handle_client_lifecycle(&state, &ctx).await;

        // 3. Probe server (embedded → direct Arc, external → TCP probe)
        probe_server(&state, &embedded_server).await;

        // 4. Probe client daemon (IPC probe via spawn_blocking)
        probe_client_daemon(&state).await;

        // 5. Check interpose status (every 30s at 2s poll = every 15 iterations)
        interpose_check_counter += 1;
        let interpose_interval = 30 / poll_interval.max(1);
        if interpose_check_counter >= interpose_interval {
            interpose_check_counter = 0;
            check_interpose(&state).await;
        }

        // 6. Detect service origins (Windows SCM) — only when status is Unknown
        detect_service_origins(&state);

        ctx.request_repaint();
        tokio::time::sleep(std::time::Duration::from_secs(poll_interval)).await;
    }

    // Cleanup: stop embedded server on exit
    if let Some(srv) = embedded_server.take() {
        let _ = srv.shutdown_tx.send(true);
        let _ = srv.task_handle.await;
        let mut st = state.lock().unwrap();
        st.server.clear_monitoring();
        st.server.status = ServiceStatus::Stopped;
    }
}

// ============================================================================
// Server lifecycle (embedded start/stop)
// ============================================================================

async fn handle_server_lifecycle(
    state: &Arc<Mutex<UiState>>,
    ctx: &egui::Context,
    embedded_server: &mut Option<EmbeddedServer>,
) {
    let (start_requested, stop_requested) = {
        let mut st = state.lock().unwrap();
        let start = st.server.start_requested;
        let stop = st.server.stop_requested;
        st.server.start_requested = false;
        st.server.stop_requested = false;
        (start, stop)
    };

    // Handle start request
    if start_requested {
        // Build server config from UI state
        let server_config = {
            let st = state.lock().unwrap();
            let cfg = &st.server.config;
            ServerConfig {
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
            }
        };
        let tokens = {
            let st = state.lock().unwrap();
            st.server.config.tokens.clone()
        };

        // Try Windows Service first, then embedded
        let started_via_scm = try_start_server_service();

        if started_via_scm {
            let mut st = state.lock().unwrap();
            st.server.origin = ServiceOrigin::WindowsService;
            st.server.status = ServiceStatus::Starting;
            st.server.start_time = Some(std::time::Instant::now());
        } else {
            // Start embedded server
            let (shutdown_tx, shutdown_rx) = watch::channel(false);
            let server = Arc::new(rgpu_server::RgpuServer::new(server_config, tokens));
            let server_for_task = server.clone();
            let task_handle = tokio::spawn(async move {
                if let Err(e) = server_for_task.run_with_shutdown(shutdown_rx).await {
                    error!("embedded server error: {}", e);
                }
            });

            info!("embedded server starting");

            {
                let gpu_infos = server.gpu_infos().to_vec();
                let mut st = state.lock().unwrap();
                st.server.origin = ServiceOrigin::Embedded;
                st.server.status = ServiceStatus::Running;
                st.server.served_gpus = gpu_infos;
            }

            *embedded_server = Some(EmbeddedServer {
                shutdown_tx,
                task_handle,
                server_ref: server,
            });

            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        ctx.request_repaint();
    }

    // Handle stop request
    if stop_requested {
        let origin = {
            let st = state.lock().unwrap();
            st.server.origin.clone()
        };

        match origin {
            ServiceOrigin::Embedded => {
                if let Some(srv) = embedded_server.take() {
                    info!("stopping embedded server");
                    let _ = srv.shutdown_tx.send(true);
                    let _ = tokio::time::timeout(
                        std::time::Duration::from_secs(15),
                        srv.task_handle,
                    )
                    .await;
                }
            }
            ServiceOrigin::WindowsService => {
                if let Err(e) = service_control::scm::stop_service("RGPU Server") {
                    warn!("failed to stop RGPU Server service: {}", e);
                }
            }
            _ => {}
        }

        let mut st = state.lock().unwrap();
        st.server.clear_monitoring();
        st.server.status = ServiceStatus::Stopped;
        st.server.origin = ServiceOrigin::NotDetected;
        ctx.request_repaint();
    }
}

// ============================================================================
// Client lifecycle (service/process start/stop)
// ============================================================================

async fn handle_client_lifecycle(
    state: &Arc<Mutex<UiState>>,
    ctx: &egui::Context,
) {
    let (start_requested, stop_requested) = {
        let mut st = state.lock().unwrap();
        let start = st.client.start_requested;
        let stop = st.client.stop_requested;
        st.client.start_requested = false;
        st.client.stop_requested = false;
        (start, stop)
    };

    if start_requested {
        // Try Windows Service first
        let started_via_scm = try_start_client_service();

        if started_via_scm {
            let mut st = state.lock().unwrap();
            st.client.origin = ServiceOrigin::WindowsService;
            st.client.status = ServiceStatus::Starting;
            st.client.start_time = Some(std::time::Instant::now());
        } else {
            // Spawn client process
            let config_path = {
                let st = state.lock().unwrap();
                st.config_path.clone()
            };

            match service_control::spawn_client_process(&config_path) {
                Ok(_child) => {
                    let mut st = state.lock().unwrap();
                    st.client.origin = ServiceOrigin::ExternalProcess;
                    st.client.status = ServiceStatus::Starting;
                    st.client.start_time = Some(std::time::Instant::now());
                    info!("spawned client daemon process (config: {})", config_path);
                }
                Err(e) => {
                    let mut st = state.lock().unwrap();
                    st.client.status = ServiceStatus::Error(format!("spawn failed: {}", e));
                    st.push_error(format!("failed to start client: {}", e));
                }
            }
        }
        ctx.request_repaint();
    }

    if stop_requested {
        let origin = {
            let st = state.lock().unwrap();
            st.client.origin.clone()
        };

        match origin {
            ServiceOrigin::WindowsService => {
                if let Err(e) = service_control::scm::stop_service("RGPU Client") {
                    warn!("failed to stop RGPU Client service: {}", e);
                }
            }
            ServiceOrigin::ExternalProcess => {
                // Send Shutdown message to the daemon via IPC
                let result = tokio::task::spawn_blocking(|| {
                    service_detection::send_shutdown_to_daemon()
                }).await;
                match result {
                    Ok(Ok(())) => info!("sent Shutdown to client daemon via IPC"),
                    Ok(Err(e)) => warn!("failed to send Shutdown to daemon: {}", e),
                    Err(e) => warn!("shutdown task failed: {}", e),
                }
            }
            _ => {}
        }

        let mut st = state.lock().unwrap();
        st.client.clear_monitoring();
        st.client.status = ServiceStatus::Stopped;
        st.client.origin = ServiceOrigin::NotDetected;
        ctx.request_repaint();
    }
}

// ============================================================================
// Server probing
// ============================================================================

async fn probe_server(
    state: &Arc<Mutex<UiState>>,
    embedded_server: &Option<EmbeddedServer>,
) {
    // If embedded, poll directly via Arc
    if let Some(ref srv) = embedded_server {
        let metrics_ref = srv.server_ref.metrics();
        let uptime = metrics_ref.start_time.elapsed().as_secs();
        let snapshot = MetricsSnapshot {
            timestamp: std::time::Instant::now(),
            connections_total: metrics_ref
                .connections_total
                .load(std::sync::atomic::Ordering::Relaxed),
            connections_active: metrics_ref
                .connections_active
                .load(std::sync::atomic::Ordering::Relaxed),
            requests_total: metrics_ref
                .requests_total
                .load(std::sync::atomic::Ordering::Relaxed),
            errors_total: metrics_ref
                .errors_total
                .load(std::sync::atomic::Ordering::Relaxed),
            cuda_commands: metrics_ref
                .cuda_commands
                .load(std::sync::atomic::Ordering::Relaxed),
            vulkan_commands: metrics_ref
                .vulkan_commands
                .load(std::sync::atomic::Ordering::Relaxed),
            uptime_secs: uptime,
        };
        let gpu_infos = srv.server_ref.gpu_infos().to_vec();

        let mut st = state.lock().unwrap();
        st.server.served_gpus = gpu_infos;
        st.server.push_metrics(snapshot);
        st.server.status = ServiceStatus::Running;
        return;
    }

    // External server: probe via TCP (blocking, run in spawn_blocking)
    let (port, token) = {
        let st = state.lock().unwrap();
        let port = st.server.config.port;
        // Get the first token for authentication
        let token = st
            .server
            .config
            .tokens
            .first()
            .map(|t| t.token.clone())
            .or_else(|| {
                st.config_editor
                    .as_ref()
                    .and_then(|e| e.config.security.tokens.first().map(|t| t.token.clone()))
            })
            .unwrap_or_default();
        (port, token)
    };

    if token.is_empty() {
        // No token configured, can't probe
        return;
    }

    let result =
        tokio::task::spawn_blocking(move || service_detection::probe_server_tcp(port, &token))
            .await;

    match result {
        Ok(Ok(probe)) => {
            let mut st = state.lock().unwrap();
            st.server.served_gpus = probe.gpus;
            if st.server.origin == ServiceOrigin::NotDetected {
                st.server.origin = ServiceOrigin::ExternalProcess;
            }
            st.server.status = ServiceStatus::Running;
            st.server.start_time = None; // Connected, clear grace period

            if let Some(m) = probe.metrics {
                let snapshot = MetricsSnapshot {
                    timestamp: std::time::Instant::now(),
                    connections_total: m.connections_total,
                    connections_active: m.connections_active,
                    requests_total: m.requests_total,
                    errors_total: m.errors_total,
                    cuda_commands: m.cuda_commands,
                    vulkan_commands: m.vulkan_commands,
                    uptime_secs: m.uptime_secs,
                };
                st.server.push_metrics(snapshot);
            }
        }
        Ok(Err(e)) => {
            // Server not reachable
            let mut st = state.lock().unwrap();
            if st.server.origin == ServiceOrigin::Embedded {
                return;
            }

            // Don't override Starting status during grace period
            if matches!(st.server.status, ServiceStatus::Starting) {
                if st.server.in_startup_grace() {
                    debug!("server TCP probe failed during startup grace period: {}", e);
                    return;
                }
                st.server.status =
                    ServiceStatus::Error("server did not respond after startup".to_string());
                st.server.start_time = None;
                return;
            }

            if st.server.status.is_running() {
                // Was running, now gone
                st.server.clear_monitoring();
                st.server.origin = ServiceOrigin::NotDetected;
            }
            st.server.status = ServiceStatus::Stopped;
        }
        Err(e) => {
            debug!("server probe task failed: {}", e);
        }
    }
}

// ============================================================================
// Client daemon IPC probing
// ============================================================================

async fn probe_client_daemon(state: &Arc<Mutex<UiState>>) {
    let result =
        tokio::task::spawn_blocking(|| service_detection::probe_client_ipc()).await;

    match result {
        Ok(Ok(gpus)) => {
            let mut st = state.lock().unwrap();
            if st.client.origin == ServiceOrigin::NotDetected {
                st.client.origin = ServiceOrigin::ExternalProcess;
            }
            st.client.status = ServiceStatus::Running;
            st.client.start_time = None; // Connected, clear grace period

            // Derive remote servers from GPU pool
            let mut servers: Vec<ClientRemoteServer> = Vec::new();
            for gpu in &gpus {
                if gpu.server_id != 0 {
                    if let Some(existing) = servers.iter_mut().find(|s| s.server_id == Some(gpu.server_id)) {
                        existing.gpu_count += 1;
                    } else {
                        servers.push(ClientRemoteServer {
                            address: format!("server-{}", gpu.server_id),
                            server_id: Some(gpu.server_id),
                            connected: true,
                            gpu_count: 1,
                        });
                    }
                }
            }
            st.client.remote_servers = servers;
            st.client.gpu_pool = gpus;
        }
        Ok(Err(e)) => {
            // Client daemon not reachable
            let mut st = state.lock().unwrap();

            // Don't override Starting status during grace period
            if matches!(st.client.status, ServiceStatus::Starting) {
                if st.client.in_startup_grace() {
                    debug!("client IPC probe failed during startup grace period: {}", e);
                    return;
                }
                // Grace period expired — daemon failed to start
                st.client.status =
                    ServiceStatus::Error("daemon did not respond after startup".to_string());
                st.client.start_time = None;
                return;
            }

            if st.client.status.is_running() && st.client.origin != ServiceOrigin::Embedded {
                st.client.clear_monitoring();
                st.client.origin = ServiceOrigin::NotDetected;
            }
            st.client.status = ServiceStatus::Stopped;
        }
        Err(e) => {
            debug!("client IPC probe task failed: {}", e);
        }
    }
}

// ============================================================================
// Interpose status checking
// ============================================================================

async fn check_interpose(state: &Arc<Mutex<UiState>>) {
    let result =
        tokio::task::spawn_blocking(|| service_detection::check_interpose_status()).await;

    if let Ok(status) = result {
        let mut st = state.lock().unwrap();
        st.client.interpose = status;
    }
}

// ============================================================================
// Windows SCM detection
// ============================================================================

fn detect_service_origins(state: &Arc<Mutex<UiState>>) {
    let mut st = state.lock().unwrap();

    // Only detect if not already known
    if st.server.origin == ServiceOrigin::NotDetected {
        if let Some(scm_status) = service_control::scm::query_service_status("RGPU Server") {
            match scm_status {
                service_control::scm::ServiceState::Running => {
                    st.server.origin = ServiceOrigin::WindowsService;
                    // Status will be set by probe
                }
                service_control::scm::ServiceState::StartPending => {
                    st.server.origin = ServiceOrigin::WindowsService;
                    st.server.status = ServiceStatus::Starting;
                }
                service_control::scm::ServiceState::StopPending => {
                    st.server.origin = ServiceOrigin::WindowsService;
                    st.server.status = ServiceStatus::Stopping;
                }
                _ => {}
            }
        }
    }

    if st.client.origin == ServiceOrigin::NotDetected {
        if let Some(scm_status) = service_control::scm::query_service_status("RGPU Client") {
            match scm_status {
                service_control::scm::ServiceState::Running => {
                    st.client.origin = ServiceOrigin::WindowsService;
                    // Status will be set by probe
                }
                service_control::scm::ServiceState::StartPending => {
                    st.client.origin = ServiceOrigin::WindowsService;
                    st.client.status = ServiceStatus::Starting;
                }
                service_control::scm::ServiceState::StopPending => {
                    st.client.origin = ServiceOrigin::WindowsService;
                    st.client.status = ServiceStatus::Stopping;
                }
                _ => {}
            }
        }
    }
}

// ============================================================================
// SCM helpers
// ============================================================================

fn try_start_server_service() -> bool {
    if service_control::scm::is_service_installed("RGPU Server") {
        match service_control::scm::start_service("RGPU Server") {
            Ok(()) => {
                info!("started RGPU Server via Windows SCM");
                return true;
            }
            Err(e) => {
                warn!("SCM start RGPU Server failed: {}", e);
            }
        }
    }
    false
}

fn try_start_client_service() -> bool {
    if service_control::scm::is_service_installed("RGPU Client") {
        match service_control::scm::start_service("RGPU Client") {
            Ok(()) => {
                info!("started RGPU Client via Windows SCM");
                return true;
            }
            Err(e) => {
                warn!("SCM start RGPU Client failed: {}", e);
            }
        }
    }
    false
}
