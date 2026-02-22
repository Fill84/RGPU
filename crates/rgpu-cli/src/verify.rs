use std::time::Duration;

use rgpu_common::platform::default_ipc_path;
use rgpu_core::config::RgpuConfig;
use rgpu_protocol::gpu_info::GpuInfo;
use rgpu_protocol::messages::{Message, PROTOCOL_VERSION};
use rgpu_protocol::wire;

/// Local GPU marker (u16::MAX) — matches rgpu_client::pool_manager::LOCAL_SERVER_ID.
const LOCAL_SERVER_ID: u16 = u16::MAX;

// ── Check result types ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum CheckStatus {
    Pass,
    Fail,
    Warn,
    Skip,
}

#[derive(Debug)]
struct CheckResult {
    name: String,
    status: CheckStatus,
    message: String,
    details: Vec<String>,
}

impl CheckResult {
    fn pass(name: &str, message: &str) -> Self {
        Self {
            name: name.to_string(),
            status: CheckStatus::Pass,
            message: message.to_string(),
            details: Vec::new(),
        }
    }

    fn fail(name: &str, message: &str) -> Self {
        Self {
            name: name.to_string(),
            status: CheckStatus::Fail,
            message: message.to_string(),
            details: Vec::new(),
        }
    }

    fn warn(name: &str, message: &str) -> Self {
        Self {
            name: name.to_string(),
            status: CheckStatus::Warn,
            message: message.to_string(),
            details: Vec::new(),
        }
    }

    fn skip(name: &str, message: &str) -> Self {
        Self {
            name: name.to_string(),
            status: CheckStatus::Skip,
            message: message.to_string(),
            details: Vec::new(),
        }
    }

    fn detail(mut self, detail: &str) -> Self {
        self.details.push(detail.to_string());
        self
    }
}

// ── Main entry point ────────────────────────────────────────────────────────

pub async fn run_verify(config_path: &str, json: bool) -> anyhow::Result<()> {
    let mut results: Vec<CheckResult> = Vec::new();

    // Check 1: Configuration
    let rgpu_config = check_config(config_path, &mut results);

    // Check 2 + 3: Client daemon + GPU pool
    let daemon_gpus = check_daemon(&mut results).await;
    check_gpu_pool(daemon_gpus.as_deref(), &mut results);

    // Check 4: Server connectivity
    if let Some(ref config) = rgpu_config {
        for endpoint in &config.client.servers {
            check_server_connectivity(endpoint, &mut results).await;
        }
    } else if rgpu_config.is_none() {
        results.push(CheckResult::skip(
            "Server connectivity",
            "No config loaded, cannot check servers",
        ));
    }

    // Check 5: CUDA interpose
    check_cuda_interpose(&mut results);

    // Check 6: Vulkan ICD
    check_vulkan_icd(&mut results);

    // Output
    if json {
        print_results_json(&results);
    } else {
        print_results_pretty(&results);
    }

    // Exit code 1 if any failures
    if results
        .iter()
        .any(|r| matches!(r.status, CheckStatus::Fail))
    {
        std::process::exit(1);
    }

    Ok(())
}

// ── Check 1: Configuration ──────────────────────────────────────────────────

fn check_config(config_path: &str, results: &mut Vec<CheckResult>) -> Option<RgpuConfig> {
    let path = std::path::Path::new(config_path);

    if !path.exists() {
        results.push(
            CheckResult::warn(
                "Configuration",
                &format!("Config file not found: {}", config_path),
            )
            .detail("Using default configuration")
            .detail("Create rgpu.toml or install via the RGPU installer"),
        );
        return None;
    }

    match RgpuConfig::load(config_path) {
        Ok(config) => {
            let mut result = CheckResult::pass(
                "Configuration",
                &format!("Loaded from {}", config_path),
            );

            if config.client.servers.is_empty() {
                result = result.detail("No remote servers configured");
            } else {
                for (i, ep) in config.client.servers.iter().enumerate() {
                    result = result.detail(&format!(
                        "Server {}: {} ({:?})",
                        i + 1,
                        ep.address,
                        ep.transport
                    ));
                }
            }

            result = result.detail(&format!(
                "Include local GPUs: {}",
                config.client.include_local_gpus
            ));
            result = result.detail(&format!("GPU ordering: {:?}", config.client.gpu_ordering));

            results.push(result);
            Some(config)
        }
        Err(e) => {
            results.push(CheckResult::fail(
                "Configuration",
                &format!("Failed to parse {}: {}", config_path, e),
            ));
            None
        }
    }
}

// ── Check 2: Client Daemon via IPC ──────────────────────────────────────────

async fn check_daemon(results: &mut Vec<CheckResult>) -> Option<Vec<GpuInfo>> {
    let ipc_path = default_ipc_path();

    let connect_result =
        tokio::task::spawn_blocking(move || try_daemon_query(&ipc_path)).await;

    match connect_result {
        Ok(Ok(gpus)) => {
            let result = CheckResult::pass(
                "Client daemon",
                &format!("Connected via {}", default_ipc_path()),
            )
            .detail(&format!("GPU pool: {} GPU(s) available", gpus.len()));
            results.push(result);
            Some(gpus)
        }
        Ok(Err(e)) => {
            let result = CheckResult::fail(
                "Client daemon",
                &format!("Cannot connect: {}", e),
            )
            .detail(&format!("IPC path: {}", default_ipc_path()))
            .detail("Is the client daemon running? Start with: rgpu client");
            results.push(result);
            None
        }
        Err(e) => {
            results.push(CheckResult::fail(
                "Client daemon",
                &format!("Internal error: {}", e),
            ));
            None
        }
    }
}

fn try_daemon_query(ipc_path: &str) -> Result<Vec<GpuInfo>, String> {
    use std::io::{Read, Write};

    // Connect to IPC
    #[cfg(unix)]
    let mut stream = {
        let s = std::os::unix::net::UnixStream::connect(ipc_path)
            .map_err(|e| format!("connection failed: {}", e))?;
        s.set_read_timeout(Some(Duration::from_secs(5))).ok();
        s.set_write_timeout(Some(Duration::from_secs(5))).ok();
        s
    };

    #[cfg(windows)]
    let mut stream = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(ipc_path)
        .map_err(|e| format!("connection failed: {}", e))?;

    // Send QueryGpus
    let msg = Message::QueryGpus;
    let frame = wire::encode_message(&msg, 0).map_err(|e| e.to_string())?;
    stream
        .write_all(&frame)
        .map_err(|e| format!("write failed: {}", e))?;

    // Read response header
    let mut header_buf = [0u8; wire::HEADER_SIZE];
    stream
        .read_exact(&mut header_buf)
        .map_err(|e| format!("read header failed: {}", e))?;

    let (flags, _, payload_len) = wire::decode_header(&header_buf).map_err(|e| e.to_string())?;

    // Read payload
    let mut payload = vec![0u8; payload_len as usize];
    stream
        .read_exact(&mut payload)
        .map_err(|e| format!("read payload failed: {}", e))?;

    let response = wire::decode_message(&payload, flags).map_err(|e| e.to_string())?;

    match response {
        Message::GpuList(gpus) => Ok(gpus),
        other => Err(format!("unexpected response: {:?}", other)),
    }
}

// ── Check 3: GPU Pool ───────────────────────────────────────────────────────

fn check_gpu_pool(gpus: Option<&[GpuInfo]>, results: &mut Vec<CheckResult>) {
    match gpus {
        None => {
            results.push(CheckResult::skip(
                "GPU pool",
                "Daemon not connected, cannot query GPU pool",
            ));
        }
        Some(gpus) if gpus.is_empty() => {
            results.push(
                CheckResult::warn("GPU pool", "No GPUs in pool")
                    .detail("Check server connectivity or enable include_local_gpus in config"),
            );
        }
        Some(gpus) => {
            let mut result =
                CheckResult::pass("GPU pool", &format!("{} GPU(s) available", gpus.len()));

            for (i, gpu) in gpus.iter().enumerate() {
                let location = if gpu.server_id == LOCAL_SERVER_ID {
                    "local"
                } else {
                    "remote"
                };
                result = result.detail(&format!(
                    "GPU {}: {} ({}, VRAM: {} MB, CUDA: {}, Vulkan: {})",
                    i,
                    gpu.device_name,
                    location,
                    gpu.total_memory / (1024 * 1024),
                    gpu.supports_cuda,
                    gpu.supports_vulkan,
                ));
            }
            results.push(result);
        }
    }
}

// ── Check 4: Server Connectivity ────────────────────────────────────────────

async fn check_server_connectivity(
    endpoint: &rgpu_core::config::ServerEndpoint,
    results: &mut Vec<CheckResult>,
) {
    let timeout = Duration::from_secs(10);
    let addr = endpoint.address.clone();
    let token = endpoint.token.clone();

    let result = tokio::time::timeout(timeout, check_server_tcp(&addr, &token)).await;

    match result {
        Ok(Ok((gpu_count, server_id))) => {
            results.push(CheckResult::pass(
                &format!("Server {}", addr),
                &format!("Connected, {} GPU(s), server_id={}", gpu_count, server_id),
            ));
        }
        Ok(Err(e)) => {
            results.push(
                CheckResult::fail(&format!("Server {}", addr), &format!("Failed: {}", e))
                    .detail("Check server address, port, and token"),
            );
        }
        Err(_) => {
            results.push(
                CheckResult::fail(&format!("Server {}", addr), "Connection timed out (10s)")
                    .detail("Server may be unreachable or firewall is blocking port"),
            );
        }
    }
}

async fn check_server_tcp(addr: &str, token: &str) -> anyhow::Result<(usize, u16)> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;

    let stream = TcpStream::connect(addr).await?;
    let (mut reader, mut writer) = stream.into_split();

    // Send Hello
    let hello = Message::Hello {
        protocol_version: PROTOCOL_VERSION,
        name: "RGPU Verify".to_string(),
        challenge: None,
    };
    let frame = wire::encode_message(&hello, 0)?;
    writer.write_all(&frame).await?;

    // Read server Hello
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

    // Authenticate
    let response = rgpu_transport::auth::compute_challenge_response(token, &challenge);
    let auth_msg = Message::Authenticate {
        token: token.to_string(),
        challenge_response: response,
    };
    let frame = wire::encode_message(&auth_msg, 0)?;
    writer.write_all(&frame).await?;

    // Read auth result
    let mut header_buf = [0u8; wire::HEADER_SIZE];
    reader.read_exact(&mut header_buf).await?;
    let (flags, _, payload_len) = wire::decode_header(&header_buf)?;
    let mut payload = vec![0u8; payload_len as usize];
    reader.read_exact(&mut payload).await?;
    let auth_result = wire::decode_message(&payload, flags)?;

    match auth_result {
        Message::AuthResult {
            success: true,
            available_gpus,
            server_id,
            ..
        } => Ok((available_gpus.len(), server_id.unwrap_or(0))),
        Message::AuthResult {
            success: false,
            error_message,
            ..
        } => anyhow::bail!(
            "authentication failed: {}",
            error_message.unwrap_or_default()
        ),
        _ => anyhow::bail!("unexpected response from server"),
    }
}

// ── Check 5: CUDA Interpose ─────────────────────────────────────────────────

fn check_cuda_interpose(results: &mut Vec<CheckResult>) {
    #[cfg(windows)]
    {
        check_cuda_interpose_windows(results);
    }
    #[cfg(not(windows))]
    {
        check_cuda_interpose_linux(results);
    }
}

#[cfg(windows)]
fn check_cuda_interpose_windows(results: &mut Vec<CheckResult>) {
    let system_root =
        std::env::var("SYSTEMROOT").unwrap_or_else(|_| r"C:\Windows".to_string());
    let nvcuda_path = format!(r"{}\System32\nvcuda.dll", system_root);
    let backup_path = format!(r"{}\System32\nvcuda_real.dll", system_root);

    if !std::path::Path::new(&nvcuda_path).exists() {
        results.push(
            CheckResult::warn("CUDA interpose", "nvcuda.dll not found in System32")
                .detail("No CUDA runtime installed or RGPU not yet installed"),
        );
        return;
    }

    // Read the DLL binary and search for our marker symbol (avoids DllMain side-effects)
    match std::fs::read(&nvcuda_path) {
        Ok(data) => {
            let marker = b"rgpu_interpose_marker";
            if data
                .windows(marker.len())
                .any(|w| w == marker.as_slice())
            {
                let mut result = CheckResult::pass(
                    "CUDA interpose",
                    "RGPU interpose DLL installed in System32",
                )
                .detail(&format!("Path: {}", nvcuda_path));

                if std::path::Path::new(&backup_path).exists() {
                    result =
                        result.detail("Original NVIDIA driver backed up as nvcuda_real.dll");
                } else {
                    result = result
                        .detail("WARNING: nvcuda_real.dll backup not found — uninstall may not restore original driver");
                }
                results.push(result);
            } else {
                results.push(
                    CheckResult::warn(
                        "CUDA interpose",
                        "nvcuda.dll is the original NVIDIA driver (not RGPU interpose)",
                    )
                    .detail("Run the RGPU installer and select 'CUDA System-Wide Interpose'"),
                );
            }
        }
        Err(e) => {
            results.push(
                CheckResult::fail(
                    "CUDA interpose",
                    &format!("Cannot read {}: {}", nvcuda_path, e),
                )
                .detail("You may need administrator privileges"),
            );
        }
    }
}

#[cfg(not(windows))]
fn check_cuda_interpose_linux(results: &mut Vec<CheckResult>) {
    let paths = [
        "/usr/lib/rgpu/librgpu_cuda_interpose.so",
        "/usr/local/lib/rgpu/librgpu_cuda_interpose.so",
    ];

    for path in &paths {
        if std::path::Path::new(path).exists() {
            // Try to load and check for marker
            match unsafe { libloading::Library::new(path) } {
                Ok(lib) => {
                    let marker: Result<
                        libloading::Symbol<extern "C" fn() -> i32>,
                        _,
                    > = unsafe { lib.get(b"rgpu_interpose_marker") };
                    if marker.is_ok() {
                        results.push(
                            CheckResult::pass(
                                "CUDA interpose",
                                &format!("Found at {}", path),
                            )
                            .detail(&format!(
                                "Use: LD_PRELOAD={} <application>",
                                path
                            )),
                        );
                        return;
                    }
                }
                Err(e) => {
                    results.push(CheckResult::warn(
                        "CUDA interpose",
                        &format!("Found at {} but cannot load: {}", path, e),
                    ));
                    return;
                }
            }
        }
    }

    results.push(
        CheckResult::warn("CUDA interpose", "CUDA interpose library not found")
            .detail("Expected at: /usr/lib/rgpu/librgpu_cuda_interpose.so")
            .detail("Install via .deb/.rpm package"),
    );
}

// ── Check 6: Vulkan ICD ─────────────────────────────────────────────────────

fn check_vulkan_icd(results: &mut Vec<CheckResult>) {
    #[cfg(windows)]
    {
        check_vulkan_icd_windows(results);
    }
    #[cfg(not(windows))]
    {
        check_vulkan_icd_linux(results);
    }
}

#[cfg(windows)]
fn check_vulkan_icd_windows(results: &mut Vec<CheckResult>) {
    use windows_sys::Win32::System::Registry::*;

    let subkey = b"SOFTWARE\\Khronos\\Vulkan\\Drivers\0";
    let mut hkey: HKEY = std::ptr::null_mut();

    let status = unsafe {
        RegOpenKeyExA(
            HKEY_LOCAL_MACHINE,
            subkey.as_ptr(),
            0,
            KEY_READ,
            &mut hkey,
        )
    };

    if status != 0 {
        results.push(
            CheckResult::warn(
                "Vulkan ICD",
                "Vulkan drivers registry key not found",
            )
            .detail("Key: HKLM\\SOFTWARE\\Khronos\\Vulkan\\Drivers")
            .detail("Vulkan runtime may not be installed"),
        );
        return;
    }

    // Enumerate registry values to find rgpu_icd.json
    let mut found = false;
    let mut found_path = String::new();
    let mut index: u32 = 0;

    loop {
        let mut name_buf = [0u8; 512];
        let mut name_len = name_buf.len() as u32;

        let status = unsafe {
            RegEnumValueA(
                hkey,
                index,
                name_buf.as_mut_ptr(),
                &mut name_len,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };

        if status != 0 {
            break;
        }

        let name = String::from_utf8_lossy(&name_buf[..name_len as usize]);
        if name.contains("rgpu_icd.json") {
            found = true;
            found_path = name.to_string();
            break;
        }

        index += 1;
    }

    unsafe {
        RegCloseKey(hkey);
    }

    if found {
        let json_exists = std::path::Path::new(&found_path).exists();
        if json_exists {
            results.push(
                CheckResult::pass("Vulkan ICD", "RGPU Vulkan ICD registered in Windows registry")
                    .detail(&format!("Manifest: {}", found_path)),
            );
        } else {
            results.push(
                CheckResult::warn(
                    "Vulkan ICD",
                    "ICD registered in registry but manifest file missing",
                )
                .detail(&format!("Registry points to: {}", found_path))
                .detail("Reinstall RGPU to fix this"),
            );
        }
    } else {
        results.push(
            CheckResult::warn("Vulkan ICD", "RGPU Vulkan ICD not found in registry")
                .detail("Run the RGPU installer and select 'Vulkan ICD Driver'"),
        );
    }
}

#[cfg(not(windows))]
fn check_vulkan_icd_linux(results: &mut Vec<CheckResult>) {
    let icd_paths = [
        "/usr/share/vulkan/icd.d/rgpu_icd.json",
        "/etc/vulkan/icd.d/rgpu_icd.json",
        "/usr/local/share/vulkan/icd.d/rgpu_icd.json",
    ];

    for path in &icd_paths {
        if std::path::Path::new(path).exists() {
            match std::fs::read_to_string(path) {
                Ok(content) => {
                    if content.contains("rgpu_vk_icd") || content.contains("librgpu_vk_icd") {
                        results.push(
                            CheckResult::pass(
                                "Vulkan ICD",
                                &format!("ICD manifest found at {}", path),
                            )
                            .detail("Vulkan loader will automatically pick up the RGPU ICD"),
                        );
                        return;
                    } else {
                        results.push(CheckResult::warn(
                            "Vulkan ICD",
                            &format!("Manifest at {} does not reference RGPU library", path),
                        ));
                        return;
                    }
                }
                Err(e) => {
                    results.push(CheckResult::warn(
                        "Vulkan ICD",
                        &format!("Cannot read {}: {}", path, e),
                    ));
                    return;
                }
            }
        }
    }

    // Also check VK_ICD_FILENAMES env var
    if let Ok(icd_env) = std::env::var("VK_ICD_FILENAMES") {
        if icd_env.contains("rgpu") {
            results.push(
                CheckResult::pass(
                    "Vulkan ICD",
                    &format!("VK_ICD_FILENAMES set to: {}", icd_env),
                )
                .detail("Vulkan loader will use this override"),
            );
            return;
        }
    }

    results.push(
        CheckResult::warn("Vulkan ICD", "RGPU Vulkan ICD manifest not found")
            .detail("Expected at: /usr/share/vulkan/icd.d/rgpu_icd.json")
            .detail("Install via .deb/.rpm package"),
    );
}

// ── Output formatters ───────────────────────────────────────────────────────

fn print_results_pretty(results: &[CheckResult]) {
    println!();
    println!("RGPU Installation Verification");
    println!("==============================");
    println!();

    let mut pass_count = 0u32;
    let mut fail_count = 0u32;
    let mut warn_count = 0u32;

    for result in results {
        let (icon, color_start, color_end) = match result.status {
            CheckStatus::Pass => {
                pass_count += 1;
                ("[PASS]", "\x1b[32m", "\x1b[0m")
            }
            CheckStatus::Fail => {
                fail_count += 1;
                ("[FAIL]", "\x1b[31m", "\x1b[0m")
            }
            CheckStatus::Warn => {
                warn_count += 1;
                ("[WARN]", "\x1b[33m", "\x1b[0m")
            }
            CheckStatus::Skip => {
                ("[SKIP]", "\x1b[90m", "\x1b[0m")
            }
        };

        println!(
            "  {}{}{} {} - {}",
            color_start, icon, color_end, result.name, result.message
        );

        for detail in &result.details {
            println!("         {}", detail);
        }
        println!();
    }

    println!("-------------------------------");
    println!(
        "  {} passed, {} failed, {} warnings",
        pass_count, fail_count, warn_count
    );
    println!();
}

fn print_results_json(results: &[CheckResult]) {
    print!("[");
    for (i, result) in results.iter().enumerate() {
        let status_str = match result.status {
            CheckStatus::Pass => "pass",
            CheckStatus::Fail => "fail",
            CheckStatus::Warn => "warn",
            CheckStatus::Skip => "skip",
        };

        // Escape strings for JSON
        let name = result.name.replace('\\', "\\\\").replace('"', "\\\"");
        let message = result.message.replace('\\', "\\\\").replace('"', "\\\"");

        print!(
            "{{\"name\":\"{}\",\"status\":\"{}\",\"message\":\"{}\"",
            name, status_str, message
        );

        if !result.details.is_empty() {
            print!(",\"details\":[");
            for (j, detail) in result.details.iter().enumerate() {
                let detail_escaped = detail.replace('\\', "\\\\").replace('"', "\\\"");
                print!("\"{}\"", detail_escaped);
                if j + 1 < result.details.len() {
                    print!(",");
                }
            }
            print!("]");
        }

        print!("}}");
        if i + 1 < results.len() {
            print!(",");
        }
    }
    println!("]");
}
