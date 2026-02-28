//! Service control for starting/stopping RGPU server and client daemon.

use tracing::info;

// ============================================================================
// Process-based service management (cross-platform)
// ============================================================================

/// Spawn the RGPU server as a child process.
pub fn spawn_server_process(config_path: &str) -> Result<std::process::Child, String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot find executable: {}", e))?;
    info!("Spawning server process: {:?} server --config {}", exe, config_path);
    std::process::Command::new(&exe)
        .arg("server")
        .arg("--config")
        .arg(config_path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("failed to spawn server process: {}", e))
}

/// Spawn the RGPU client daemon as a child process.
pub fn spawn_client_process(config_path: &str) -> Result<std::process::Child, String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot find executable: {}", e))?;
    info!("Spawning client process: {:?} client --config {}", exe, config_path);
    std::process::Command::new(&exe)
        .arg("client")
        .arg("--config")
        .arg(config_path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("failed to spawn client process: {}", e))
}

// ============================================================================
// Windows Service Control Manager (SCM) integration
// ============================================================================

#[cfg(windows)]
pub mod scm {
    use tracing::{info, warn};
    use windows_sys::Win32::Foundation::{
        GetLastError, ERROR_ACCESS_DENIED, ERROR_SERVICE_ALREADY_RUNNING,
        ERROR_SERVICE_NOT_ACTIVE,
    };
    use windows_sys::Win32::System::Services::*;

    pub const SERVER_SERVICE_NAME: &str = "RGPU Server";
    pub const CLIENT_SERVICE_NAME: &str = "RGPU Client";

    /// Possible states of a Windows service.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ServiceState {
        Stopped,
        StartPending,
        Running,
        StopPending,
        Unknown,
    }

    impl std::fmt::Display for ServiceState {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ServiceState::Stopped => write!(f, "Stopped"),
                ServiceState::StartPending => write!(f, "Start Pending"),
                ServiceState::Running => write!(f, "Running"),
                ServiceState::StopPending => write!(f, "Stop Pending"),
                ServiceState::Unknown => write!(f, "Unknown"),
            }
        }
    }

    /// Convert a Rust string to a wide null-terminated string for Windows APIs.
    fn to_wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    type ScHandle = *mut std::ffi::c_void;

    /// Open the Service Control Manager with the specified access rights.
    /// Returns the SCM handle, or null on failure.
    unsafe fn open_scm(desired_access: u32) -> ScHandle {
        let handle = OpenSCManagerW(std::ptr::null(), std::ptr::null(), desired_access);
        if handle.is_null() {
            let err = GetLastError();
            if err == ERROR_ACCESS_DENIED {
                warn!("Access denied opening Service Control Manager (run as Administrator)");
            } else {
                warn!("Failed to open Service Control Manager, error code: {}", err);
            }
        }
        handle
    }

    /// Open a service by name with the specified access rights.
    /// Returns the service handle, or null on failure. Caller must close the handle.
    unsafe fn open_service_handle(scm: ScHandle, service_name: &str, desired_access: u32) -> ScHandle {
        let wide_name = to_wide(service_name);
        let handle = OpenServiceW(scm, wide_name.as_ptr(), desired_access);
        if handle.is_null() {
            let err = GetLastError();
            if err == ERROR_ACCESS_DENIED {
                warn!(
                    "Access denied opening service '{}' (run as Administrator)",
                    service_name
                );
            }
            // Other errors (e.g. service not found) are silently ignored
        }
        handle
    }

    /// Query the current status of a Windows service.
    ///
    /// Returns `None` if the service cannot be queried (not installed, access denied, etc.).
    pub fn query_service_status(service_name: &str) -> Option<ServiceState> {
        unsafe {
            let scm = open_scm(SC_MANAGER_CONNECT);
            if scm.is_null() {
                return None;
            }

            let svc = open_service_handle(scm, service_name, SERVICE_QUERY_STATUS);
            if svc.is_null() {
                CloseServiceHandle(scm);
                return None;
            }

            let mut status: SERVICE_STATUS = std::mem::zeroed();
            let result = QueryServiceStatus(svc, &mut status);

            CloseServiceHandle(svc);
            CloseServiceHandle(scm);

            if result == 0 {
                let err = GetLastError();
                warn!(
                    "Failed to query service '{}' status, error code: {}",
                    service_name, err
                );
                return None;
            }

            let state = match status.dwCurrentState {
                SERVICE_STOPPED => ServiceState::Stopped,
                SERVICE_START_PENDING => ServiceState::StartPending,
                SERVICE_STOP_PENDING => ServiceState::StopPending,
                SERVICE_RUNNING => ServiceState::Running,
                _ => ServiceState::Unknown,
            };

            Some(state)
        }
    }

    /// Start a Windows service.
    ///
    /// Returns `Ok(())` if the service was started (or was already running).
    /// Returns `Err` with a description on failure.
    pub fn start_service(service_name: &str) -> Result<(), String> {
        unsafe {
            let scm = open_scm(SC_MANAGER_CONNECT);
            if scm.is_null() {
                return Err(format!(
                    "Cannot open Service Control Manager to start '{}'",
                    service_name
                ));
            }

            let svc = open_service_handle(scm, service_name, SERVICE_START | SERVICE_QUERY_STATUS);
            if svc.is_null() {
                let err = GetLastError();
                CloseServiceHandle(scm);
                if err == ERROR_ACCESS_DENIED {
                    return Err(format!(
                        "Access denied starting service '{}' (run as Administrator)",
                        service_name
                    ));
                }
                return Err(format!(
                    "Cannot open service '{}', error code: {}",
                    service_name, err
                ));
            }

            let result = StartServiceW(svc, 0, std::ptr::null());
            if result == 0 {
                let err = GetLastError();
                CloseServiceHandle(svc);
                CloseServiceHandle(scm);

                if err == ERROR_SERVICE_ALREADY_RUNNING {
                    info!("Service '{}' is already running", service_name);
                    return Ok(());
                }
                if err == ERROR_ACCESS_DENIED {
                    return Err(format!(
                        "Access denied starting service '{}' (run as Administrator)",
                        service_name
                    ));
                }
                return Err(format!(
                    "Failed to start service '{}', error code: {}",
                    service_name, err
                ));
            }

            info!("Service '{}' start requested successfully", service_name);
            CloseServiceHandle(svc);
            CloseServiceHandle(scm);
            Ok(())
        }
    }

    /// Stop a Windows service.
    ///
    /// Returns `Ok(())` if the service was stopped (or was already stopped).
    /// Returns `Err` with a description on failure.
    pub fn stop_service(service_name: &str) -> Result<(), String> {
        unsafe {
            let scm = open_scm(SC_MANAGER_CONNECT);
            if scm.is_null() {
                return Err(format!(
                    "Cannot open Service Control Manager to stop '{}'",
                    service_name
                ));
            }

            let svc = open_service_handle(scm, service_name, SERVICE_STOP | SERVICE_QUERY_STATUS);
            if svc.is_null() {
                let err = GetLastError();
                CloseServiceHandle(scm);
                if err == ERROR_ACCESS_DENIED {
                    return Err(format!(
                        "Access denied stopping service '{}' (run as Administrator)",
                        service_name
                    ));
                }
                return Err(format!(
                    "Cannot open service '{}', error code: {}",
                    service_name, err
                ));
            }

            let mut status: SERVICE_STATUS = std::mem::zeroed();
            let result = ControlService(svc, SERVICE_CONTROL_STOP, &mut status);
            if result == 0 {
                let err = GetLastError();
                CloseServiceHandle(svc);
                CloseServiceHandle(scm);

                if err == ERROR_SERVICE_NOT_ACTIVE {
                    info!("Service '{}' is already stopped", service_name);
                    return Ok(());
                }
                if err == ERROR_ACCESS_DENIED {
                    return Err(format!(
                        "Access denied stopping service '{}' (run as Administrator)",
                        service_name
                    ));
                }
                return Err(format!(
                    "Failed to stop service '{}', error code: {}",
                    service_name, err
                ));
            }

            info!("Service '{}' stop requested successfully", service_name);
            CloseServiceHandle(svc);
            CloseServiceHandle(scm);
            Ok(())
        }
    }

    /// Check if a service is installed on the system.
    ///
    /// Returns `true` if the service exists (regardless of its current state).
    pub fn is_service_installed(service_name: &str) -> bool {
        unsafe {
            let scm = open_scm(SC_MANAGER_CONNECT);
            if scm.is_null() {
                return false;
            }

            let svc = open_service_handle(scm, service_name, SERVICE_QUERY_STATUS);
            let installed = !svc.is_null();
            if installed {
                CloseServiceHandle(svc);
            }
            CloseServiceHandle(scm);
            installed
        }
    }
}

#[cfg(not(windows))]
pub mod scm {
    pub const SERVER_SERVICE_NAME: &str = "rgpu-server";
    pub const CLIENT_SERVICE_NAME: &str = "rgpu-client";

    /// Possible states of a service (stub on non-Windows platforms).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ServiceState {
        Stopped,
        StartPending,
        Running,
        StopPending,
        Unknown,
    }

    impl std::fmt::Display for ServiceState {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ServiceState::Stopped => write!(f, "Stopped"),
                ServiceState::StartPending => write!(f, "Start Pending"),
                ServiceState::Running => write!(f, "Running"),
                ServiceState::StopPending => write!(f, "Stop Pending"),
                ServiceState::Unknown => write!(f, "Unknown"),
            }
        }
    }

    /// Query the current status of a service (not available on this platform).
    pub fn query_service_status(_service_name: &str) -> Option<ServiceState> {
        None
    }

    /// Start a service (not available on this platform).
    pub fn start_service(_service_name: &str) -> Result<(), String> {
        Err("Windows service management not available on this platform".to_string())
    }

    /// Stop a service (not available on this platform).
    pub fn stop_service(_service_name: &str) -> Result<(), String> {
        Err("Windows service management not available on this platform".to_string())
    }

    /// Check if a service is installed (always returns false on non-Windows platforms).
    pub fn is_service_installed(_service_name: &str) -> bool {
        false
    }
}
