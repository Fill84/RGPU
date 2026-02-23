//! Windows Service Control Manager (SCM) integration.
//!
//! Provides entry points for running rgpu server and client as Windows services.
//! When the NSIS installer creates services with `--service`, this module handles
//! the SCM lifecycle: registering, reporting status, and handling stop signals.

use std::ffi::OsString;
use std::time::Duration;

use tracing::info;
use windows_service::service::{
    ServiceControl, ServiceControlAccept, ServiceExitCode, ServiceState, ServiceStatus,
    ServiceType,
};
use windows_service::service_control_handler::{self, ServiceControlHandlerResult};
use windows_service::{define_windows_service, service_dispatcher};

const SERVER_SERVICE_NAME: &str = "RGPU Server";
const CLIENT_SERVICE_NAME: &str = "RGPU Client";

// ── Server Service ───────────────────────────────────────────────────

define_windows_service!(ffi_server_service_main, server_service_main);

pub fn run_as_server_service() -> anyhow::Result<()> {
    service_dispatcher::start(SERVER_SERVICE_NAME, ffi_server_service_main)
        .map_err(|e| anyhow::anyhow!("failed to start server service dispatcher: {}", e))
}

fn server_service_main(_arguments: Vec<OsString>) {
    if let Err(e) = run_server_service_inner() {
        tracing::error!("server service failed: {}", e);
    }
}

fn run_server_service_inner() -> anyhow::Result<()> {
    rgpu_common::init_logging();

    // Create a shutdown channel
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    // Register the service control handler
    let shutdown_signal = shutdown_tx.clone();
    let event_handler = move |control_event| -> ServiceControlHandlerResult {
        match control_event {
            ServiceControl::Stop | ServiceControl::Shutdown => {
                info!("service stop requested");
                let _ = shutdown_signal.send(true);
                ServiceControlHandlerResult::NoError
            }
            ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
            _ => ServiceControlHandlerResult::NotImplemented,
        }
    };

    let status_handle =
        service_control_handler::register(SERVER_SERVICE_NAME, event_handler)
            .map_err(|e| anyhow::anyhow!("failed to register service handler: {}", e))?;

    // Report: Starting
    status_handle
        .set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: ServiceState::StartPending,
            controls_accepted: ServiceControlAccept::empty(),
            exit_code: ServiceExitCode::Win32(0),
            checkpoint: 0,
            wait_hint: Duration::from_secs(10),
            process_id: None,
        })
        .map_err(|e| anyhow::anyhow!("failed to set start pending: {}", e))?;

    // Parse config from the original command line (binPath args)
    let config_path = parse_config_from_args();
    let config = config_path.unwrap_or_else(rgpu_core::config::default_config_path);

    info!("server service starting (config: {})", config);

    let rgpu_config = rgpu_core::config::RgpuConfig::load_or_default(&config);
    let server_config = rgpu_config.server.clone();

    let server =
        rgpu_server::RgpuServer::new(server_config, rgpu_config.security.tokens.clone());

    // Report: Running
    status_handle
        .set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: ServiceState::Running,
            controls_accepted: ServiceControlAccept::STOP | ServiceControlAccept::SHUTDOWN,
            exit_code: ServiceExitCode::Win32(0),
            checkpoint: 0,
            wait_hint: Duration::default(),
            process_id: None,
        })
        .map_err(|e| anyhow::anyhow!("failed to set running: {}", e))?;

    // Build a tokio runtime and run the server
    let rt = tokio::runtime::Runtime::new()?;
    let result = rt.block_on(async {
        server
            .run_with_shutdown(shutdown_rx)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))
    });

    if let Err(ref e) = result {
        tracing::error!("server error: {}", e);
    }

    // Report: Stopped
    let exit_code = if result.is_ok() {
        ServiceExitCode::Win32(0)
    } else {
        ServiceExitCode::Win32(1)
    };

    let _ = status_handle.set_service_status(ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: ServiceState::Stopped,
        controls_accepted: ServiceControlAccept::empty(),
        exit_code,
        checkpoint: 0,
        wait_hint: Duration::default(),
        process_id: None,
    });

    info!("server service stopped");
    result
}

// ── Client Service ───────────────────────────────────────────────────

define_windows_service!(ffi_client_service_main, client_service_main);

pub fn run_as_client_service() -> anyhow::Result<()> {
    service_dispatcher::start(CLIENT_SERVICE_NAME, ffi_client_service_main)
        .map_err(|e| anyhow::anyhow!("failed to start client service dispatcher: {}", e))
}

fn client_service_main(_arguments: Vec<OsString>) {
    if let Err(e) = run_client_service_inner() {
        tracing::error!("client service failed: {}", e);
    }
}

fn run_client_service_inner() -> anyhow::Result<()> {
    rgpu_common::init_logging();

    // Create a shutdown flag
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Register the service control handler
    let stop_clone = stop_flag.clone();
    let event_handler = move |control_event| -> ServiceControlHandlerResult {
        match control_event {
            ServiceControl::Stop | ServiceControl::Shutdown => {
                info!("service stop requested");
                stop_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                ServiceControlHandlerResult::NoError
            }
            ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
            _ => ServiceControlHandlerResult::NotImplemented,
        }
    };

    let status_handle =
        service_control_handler::register(CLIENT_SERVICE_NAME, event_handler)
            .map_err(|e| anyhow::anyhow!("failed to register service handler: {}", e))?;

    // Report: Starting
    status_handle
        .set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: ServiceState::StartPending,
            controls_accepted: ServiceControlAccept::empty(),
            exit_code: ServiceExitCode::Win32(0),
            checkpoint: 0,
            wait_hint: Duration::from_secs(10),
            process_id: None,
        })
        .map_err(|e| anyhow::anyhow!("failed to set start pending: {}", e))?;

    // Parse config from the original command line
    let config_path = parse_config_from_args();
    let config = config_path.unwrap_or_else(rgpu_core::config::default_config_path);

    info!("client service starting (config: {})", config);

    let rgpu_config = rgpu_core::config::RgpuConfig::load_or_default(&config);
    let client_config = rgpu_config.client.clone();

    if client_config.servers.is_empty() && !client_config.include_local_gpus {
        let _ = status_handle.set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: ServiceState::Stopped,
            controls_accepted: ServiceControlAccept::empty(),
            exit_code: ServiceExitCode::Win32(1),
            checkpoint: 0,
            wait_hint: Duration::default(),
            process_id: None,
        });
        return Err(anyhow::anyhow!(
            "no servers configured and include_local_gpus is false"
        ));
    }

    let daemon = rgpu_client::ClientDaemon::new(client_config);

    // Report: Running
    status_handle
        .set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: ServiceState::Running,
            controls_accepted: ServiceControlAccept::STOP | ServiceControlAccept::SHUTDOWN,
            exit_code: ServiceExitCode::Win32(0),
            checkpoint: 0,
            wait_hint: Duration::default(),
            process_id: None,
        })
        .map_err(|e| anyhow::anyhow!("failed to set running: {}", e))?;

    // Build a tokio runtime and run the client daemon
    let rt = tokio::runtime::Runtime::new()?;
    let result = rt.block_on(async {
        // Run the daemon until stop is signaled
        tokio::select! {
            res = daemon.run() => {
                res.map_err(|e| anyhow::anyhow!("{}", e))
            }
            _ = async {
                loop {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    if stop_flag.load(std::sync::atomic::Ordering::SeqCst) {
                        break;
                    }
                }
            } => {
                info!("client daemon stopping via service control");
                Ok(())
            }
        }
    });

    if let Err(ref e) = result {
        tracing::error!("client error: {}", e);
    }

    // Report: Stopped
    let exit_code = if result.is_ok() {
        ServiceExitCode::Win32(0)
    } else {
        ServiceExitCode::Win32(1)
    };

    let _ = status_handle.set_service_status(ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: ServiceState::Stopped,
        controls_accepted: ServiceControlAccept::empty(),
        exit_code,
        checkpoint: 0,
        wait_hint: Duration::default(),
        process_id: None,
    });

    info!("client service stopped");
    result
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Parse the --config value from the process command line.
/// When running as a service, SCM launches the binary with the full binPath,
/// so std::env::args() contains the original arguments.
fn parse_config_from_args() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    for (i, arg) in args.iter().enumerate() {
        if arg == "--config" || arg == "-c" {
            return args.get(i + 1).cloned();
        }
        if let Some(value) = arg.strip_prefix("--config=") {
            return Some(value.to_string());
        }
    }
    None
}
