//! Service detection — probe running RGPU services (server and client daemon).
//!
//! All functions in this module are **blocking** (synchronous) and are designed
//! to be called from `tokio::task::spawn_blocking`.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use rgpu_common::platform::default_ipc_path;
use rgpu_protocol::gpu_info::GpuInfo;
use rgpu_protocol::messages::{Message, PROTOCOL_VERSION};
use rgpu_protocol::wire;
use rgpu_transport::auth::compute_challenge_response;

use crate::state::InterposeStatus;

// ============================================================================
// Server probe
// ============================================================================

/// Result of probing a running RGPU server over TCP.
pub struct ServerProbeResult {
    pub gpus: Vec<GpuInfo>,
    pub server_id: Option<u16>,
    pub metrics: Option<MetricsData>,
}

/// Metrics snapshot from a server probe.
pub struct MetricsData {
    pub connections_total: u64,
    pub connections_active: u32,
    pub requests_total: u64,
    pub errors_total: u64,
    pub cuda_commands: u64,
    pub vulkan_commands: u64,
    pub uptime_secs: u64,
}

/// Connect to an RGPU server on `127.0.0.1:{port}` via TCP, perform
/// Hello/Auth handshake, then query GPUs and metrics.
pub fn probe_server_tcp(port: u16, token: &str) -> Result<ServerProbeResult, String> {
    let addr = format!("127.0.0.1:{}", port);
    let timeout = Duration::from_secs(2);

    // Connect with timeout
    let addr_parsed: std::net::SocketAddr = addr
        .parse()
        .map_err(|e| format!("invalid address {}: {}", addr, e))?;
    let mut stream = TcpStream::connect_timeout(&addr_parsed, timeout)
        .map_err(|e| format!("TCP connect to {}: {}", addr, e))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| format!("set read timeout: {}", e))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| format!("set write timeout: {}", e))?;

    // --- Hello/Auth handshake ---

    // Send Hello
    let hello = Message::Hello {
        protocol_version: PROTOCOL_VERSION,
        name: "RGPU UI".to_string(),
        challenge: None,
    };
    send_message(&mut stream, &hello)?;

    // Read server Hello (contains challenge)
    let server_hello = read_message(&mut stream)?;
    let challenge = match &server_hello {
        Message::Hello { challenge, .. } => challenge.clone().unwrap_or_default(),
        other => {
            return Err(format!(
                "expected Hello from server, got {:?}",
                std::mem::discriminant(other)
            ));
        }
    };

    // Send Authenticate
    let challenge_response = compute_challenge_response(token, &challenge);
    let auth_msg = Message::Authenticate {
        token: token.to_string(),
        challenge_response,
    };
    send_message(&mut stream, &auth_msg)?;

    // Read AuthResult
    let auth_result = read_message(&mut stream)?;
    let server_id = match &auth_result {
        Message::AuthResult {
            success: true,
            server_id,
            ..
        } => *server_id,
        Message::AuthResult {
            success: false,
            error_message,
            ..
        } => {
            return Err(format!(
                "authentication failed: {}",
                error_message.as_deref().unwrap_or("unknown error")
            ));
        }
        other => {
            return Err(format!(
                "expected AuthResult, got {:?}",
                std::mem::discriminant(other)
            ));
        }
    };

    // --- Query GPUs ---

    send_message(&mut stream, &Message::QueryGpus)?;
    let gpu_response = read_message(&mut stream)?;
    let gpus = match gpu_response {
        Message::GpuList(gpus) => gpus,
        other => {
            return Err(format!(
                "expected GpuList, got {:?}",
                std::mem::discriminant(&other)
            ));
        }
    };

    // --- Query Metrics (optional, server may not support it) ---

    let metrics = match query_metrics(&mut stream) {
        Ok(m) => Some(m),
        Err(_) => None,
    };

    Ok(ServerProbeResult {
        gpus,
        server_id,
        metrics,
    })
}

/// Send QueryMetrics and parse the response.
fn query_metrics(stream: &mut TcpStream) -> Result<MetricsData, String> {
    send_message(stream, &Message::QueryMetrics)?;
    let response = read_message(stream)?;
    match response {
        Message::MetricsData {
            connections_total,
            connections_active,
            requests_total,
            errors_total,
            cuda_commands,
            vulkan_commands,
            uptime_secs,
            ..
        } => Ok(MetricsData {
            connections_total,
            connections_active,
            requests_total,
            errors_total,
            cuda_commands,
            vulkan_commands,
            uptime_secs,
        }),
        other => Err(format!(
            "expected MetricsData, got {:?}",
            std::mem::discriminant(&other)
        )),
    }
}

// ============================================================================
// Client daemon IPC probe
// ============================================================================

/// Connect to the client daemon's IPC endpoint and query available GPUs.
pub fn probe_client_ipc() -> Result<Vec<GpuInfo>, String> {
    let ipc_path = default_ipc_path();

    #[cfg(windows)]
    let mut stream = {
        // On Windows, the IPC path is a named pipe (\\.\pipe\rgpu).
        // Open it as a regular file with read+write access.
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&ipc_path)
            .map_err(|e| format!("open named pipe {}: {}", ipc_path, e))?;
        file
    };

    #[cfg(unix)]
    let mut stream = {
        use std::os::unix::net::UnixStream;
        let timeout = Duration::from_secs(2);
        let s = UnixStream::connect(&ipc_path)
            .map_err(|e| format!("connect to Unix socket {}: {}", ipc_path, e))?;
        s.set_read_timeout(Some(timeout))
            .map_err(|e| format!("set read timeout: {}", e))?;
        s.set_write_timeout(Some(timeout))
            .map_err(|e| format!("set write timeout: {}", e))?;
        s
    };

    // Send QueryGpus
    let frame = wire::encode_message(&Message::QueryGpus, 0)
        .map_err(|e| format!("encode QueryGpus: {}", e))?;
    stream
        .write_all(&frame)
        .map_err(|e| format!("write to IPC: {}", e))?;

    // Read response
    let mut header_buf = [0u8; wire::HEADER_SIZE];
    stream
        .read_exact(&mut header_buf)
        .map_err(|e| format!("read IPC header: {}", e))?;
    let (flags, _stream_id, payload_len) =
        wire::decode_header(&header_buf).map_err(|e| format!("decode IPC header: {}", e))?;
    let mut payload = vec![0u8; payload_len as usize];
    stream
        .read_exact(&mut payload)
        .map_err(|e| format!("read IPC payload: {}", e))?;
    let response =
        wire::decode_message(&payload, flags).map_err(|e| format!("decode IPC message: {}", e))?;

    match response {
        Message::GpuList(gpus) => Ok(gpus),
        other => Err(format!(
            "expected GpuList from daemon, got {:?}",
            std::mem::discriminant(&other)
        )),
    }
}

// ============================================================================
// Client daemon shutdown via IPC
// ============================================================================

/// Send a Shutdown message to the client daemon via IPC.
/// Returns Ok(()) if the daemon acknowledged, Err if it couldn't be reached.
pub fn send_shutdown_to_daemon() -> Result<(), String> {
    let ipc_path = default_ipc_path();

    #[cfg(windows)]
    let mut stream = {
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&ipc_path)
            .map_err(|e| format!("open named pipe {}: {}", ipc_path, e))?
    };

    #[cfg(unix)]
    let mut stream = {
        use std::os::unix::net::UnixStream;
        let timeout = Duration::from_secs(2);
        let s = UnixStream::connect(&ipc_path)
            .map_err(|e| format!("connect to Unix socket {}: {}", ipc_path, e))?;
        s.set_read_timeout(Some(timeout))
            .map_err(|e| format!("set read timeout: {}", e))?;
        s.set_write_timeout(Some(timeout))
            .map_err(|e| format!("set write timeout: {}", e))?;
        s
    };

    // Send Shutdown
    let frame = wire::encode_message(&Message::Shutdown, 0)
        .map_err(|e| format!("encode Shutdown: {}", e))?;
    stream
        .write_all(&frame)
        .map_err(|e| format!("write to IPC: {}", e))?;

    // Read acknowledgement (best-effort, daemon may exit before responding)
    let mut header_buf = [0u8; wire::HEADER_SIZE];
    match stream.read_exact(&mut header_buf) {
        Ok(_) => {
            let (flags, _stream_id, payload_len) =
                wire::decode_header(&header_buf).map_err(|e| format!("decode header: {}", e))?;
            let mut payload = vec![0u8; payload_len as usize];
            let _ = stream.read_exact(&mut payload);
            let _ = wire::decode_message(&payload, flags);
        }
        Err(_) => {
            // Daemon may have already exited — that's fine
        }
    }

    Ok(())
}

// ============================================================================
// Interpose library detection
// ============================================================================

/// Check whether RGPU interpose libraries are installed on this system.
pub fn check_interpose_status() -> InterposeStatus {
    #[cfg(target_os = "windows")]
    {
        check_interpose_windows()
    }
    #[cfg(target_os = "linux")]
    {
        check_interpose_linux()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        // macOS or other — not supported yet, return all unknown
        InterposeStatus::default()
    }
}

#[cfg(target_os = "windows")]
fn check_interpose_windows() -> InterposeStatus {
    let cuda = Some(file_contains_marker(r"C:\Windows\System32\nvcuda.dll"));
    let nvenc = Some(file_contains_marker(
        r"C:\Windows\System32\nvEncodeAPI64.dll",
    ));
    let nvdec = Some(file_contains_marker(r"C:\Windows\System32\nvcuvid.dll"));
    let nvml = Some(file_contains_marker(r"C:\Windows\System32\nvml.dll"));
    let vulkan = Some(check_vulkan_registry());

    InterposeStatus {
        cuda,
        vulkan,
        nvenc,
        nvdec,
        nvml,
    }
}

#[cfg(target_os = "windows")]
fn file_contains_marker(path: &str) -> bool {
    let Ok(data) = std::fs::read(path) else {
        return false;
    };
    // Search for the marker bytes in the file
    let marker = b"rgpu_interpose_marker";
    data.windows(marker.len()).any(|w| w == marker)
}

#[cfg(target_os = "windows")]
fn check_vulkan_registry() -> bool {
    use windows_sys::Win32::System::Registry::*;

    const KEY_PATH: &[u8] = b"SOFTWARE\\Khronos\\Vulkan\\Drivers\0";

    unsafe {
        let mut hkey: windows_sys::Win32::System::Registry::HKEY = std::ptr::null_mut();
        let status = RegOpenKeyExA(
            HKEY_LOCAL_MACHINE,
            KEY_PATH.as_ptr(),
            0,
            KEY_READ,
            &mut hkey,
        );
        if status != 0 {
            return false;
        }

        // Enumerate values looking for one containing "rgpu"
        let mut index: u32 = 0;
        let mut found = false;
        loop {
            let mut name_buf = [0u8; 512];
            let mut name_len: u32 = name_buf.len() as u32;
            let status = RegEnumValueA(
                hkey,
                index,
                name_buf.as_mut_ptr(),
                &mut name_len,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            if status != 0 {
                break;
            }
            let name = std::str::from_utf8(&name_buf[..name_len as usize]).unwrap_or("");
            if name.to_lowercase().contains("rgpu") {
                found = true;
                break;
            }
            index += 1;
        }

        RegCloseKey(hkey);
        found
    }
}

#[cfg(target_os = "linux")]
fn check_interpose_linux() -> InterposeStatus {
    use std::path::Path;

    let cuda = Some(Path::new("/usr/lib/rgpu/librgpu_cuda_interpose.so").exists());
    let nvenc = Some(Path::new("/usr/lib/rgpu/librgpu_nvenc_interpose.so").exists());
    let nvdec = Some(Path::new("/usr/lib/rgpu/librgpu_nvdec_interpose.so").exists());
    let nvml = Some(Path::new("/usr/lib/rgpu/librgpu_nvml_interpose.so").exists());
    let vulkan = Some(
        Path::new("/usr/local/share/vulkan/icd.d/rgpu_icd.json").exists()
            || Path::new("/etc/vulkan/icd.d/rgpu_icd.json").exists(),
    );

    InterposeStatus {
        cuda,
        vulkan,
        nvenc,
        nvdec,
        nvml,
    }
}

// ============================================================================
// Wire protocol helpers (blocking I/O)
// ============================================================================

/// Encode and send a message over a blocking stream.
fn send_message(stream: &mut impl Write, msg: &Message) -> Result<(), String> {
    let frame =
        wire::encode_message(msg, 0).map_err(|e| format!("encode message: {}", e))?;
    stream
        .write_all(&frame)
        .map_err(|e| format!("write message: {}", e))
}

/// Read a single message from a blocking stream.
fn read_message(stream: &mut impl Read) -> Result<Message, String> {
    let mut header_buf = [0u8; wire::HEADER_SIZE];
    stream
        .read_exact(&mut header_buf)
        .map_err(|e| format!("read header: {}", e))?;
    let (flags, _stream_id, payload_len) =
        wire::decode_header(&header_buf).map_err(|e| format!("decode header: {}", e))?;
    let mut payload = vec![0u8; payload_len as usize];
    stream
        .read_exact(&mut payload)
        .map_err(|e| format!("read payload ({} bytes): {}", payload_len, e))?;
    wire::decode_message(&payload, flags).map_err(|e| format!("decode message: {}", e))
}
