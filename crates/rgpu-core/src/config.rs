use serde::{Deserialize, Serialize};

/// Top-level RGPU configuration, loaded from rgpu.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RgpuConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub client: ClientConfig,
    #[serde(default)]
    pub security: SecurityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Unique server identifier for multi-server routing
    #[serde(default)]
    pub server_id: u16,
    /// Port to listen on
    #[serde(default = "default_port")]
    pub port: u16,
    /// Bind address
    #[serde(default = "default_bind")]
    pub bind: String,
    /// Transport mode: "tcp" or "quic"
    #[serde(default)]
    pub transport: TransportMode,
    /// TLS certificate path
    pub cert_path: Option<String>,
    /// TLS private key path
    pub key_path: Option<String>,
    /// Which GPUs to expose (None = all)
    pub expose_gpus: Option<Vec<u32>>,
    /// Maximum clients
    #[serde(default = "default_max_clients")]
    pub max_clients: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    /// Servers to connect to
    #[serde(default)]
    pub servers: Vec<ServerEndpoint>,
    /// Include local GPUs in the pool
    #[serde(default = "default_true")]
    pub include_local_gpus: bool,
    /// GPU ordering preference
    #[serde(default)]
    pub gpu_ordering: GpuOrdering,
    /// TCP address for IPC listener (for Docker/container support).
    /// When set, the daemon listens on this TCP address alongside the
    /// local Unix socket / named pipe. Example: "0.0.0.0:9877"
    #[serde(default)]
    pub ipc_listen_address: Option<String>,
    /// Create virtual GPU device nodes in the OS for remote GPUs.
    /// When true, remote GPUs appear under "Display adapters" in Device Manager
    /// (Windows) or /dev/rgpu_gpuN (Linux).
    #[serde(default = "default_true")]
    pub create_virtual_devices: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerEndpoint {
    /// Server address (host:port)
    pub address: String,
    /// Authentication token
    pub token: String,
    /// Custom CA certificate for TLS (optional)
    pub ca_cert: Option<String>,
    /// Per-server transport override
    #[serde(default)]
    pub transport: TransportMode,
}

/// Transport protocol selection.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum TransportMode {
    /// Plain TCP or TCP+TLS (default)
    #[default]
    #[serde(rename = "tcp")]
    Tcp,
    /// QUIC (always encrypted, requires cert/key)
    #[serde(rename = "quic")]
    Quic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Accepted authentication tokens
    #[serde(default)]
    pub tokens: Vec<TokenEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEntry {
    /// The token string
    pub token: String,
    /// Human-readable name for this client
    pub name: String,
    /// Which GPUs this token can access (None = all)
    pub allowed_gpus: Option<Vec<u32>>,
    /// Memory limit in bytes (None = unlimited)
    pub max_memory: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuOrdering {
    #[default]
    LocalFirst,
    RemoteFirst,
    ByCapability,
}

impl Default for RgpuConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            client: ClientConfig::default(),
            security: SecurityConfig::default(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server_id: 0,
            port: default_port(),
            bind: default_bind(),
            transport: TransportMode::default(),
            cert_path: None,
            key_path: None,
            expose_gpus: None,
            max_clients: default_max_clients(),
        }
    }
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            servers: Vec::new(),
            include_local_gpus: true,
            gpu_ordering: GpuOrdering::default(),
            ipc_listen_address: None,
            create_virtual_devices: true,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            tokens: Vec::new(),
        }
    }
}

impl RgpuConfig {
    /// Load configuration from a TOML file.
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: RgpuConfig = toml::from_str(&content)?;
        if let Err(errors) = config.validate() {
            return Err(format!("config validation failed:\n  {}", errors.join("\n  ")).into());
        }
        Ok(config)
    }

    /// Load configuration from file if it exists, otherwise return defaults.
    pub fn load_or_default(path: &str) -> Self {
        Self::load(path).unwrap_or_default()
    }

    /// Validate the configuration, returning errors for invalid values.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Server validation
        if self.server.port == 0 {
            errors.push("server.port must be > 0".to_string());
        }
        if self.server.bind.is_empty() {
            errors.push("server.bind must not be empty".to_string());
        }
        if self.server.bind.parse::<std::net::IpAddr>().is_err()
            && self.server.bind != "0.0.0.0"
            && self.server.bind != "localhost"
        {
            errors.push(format!(
                "server.bind '{}' is not a valid IP address",
                self.server.bind
            ));
        }
        if self.server.max_clients == 0 {
            errors.push("server.max_clients must be > 0".to_string());
        }
        if self.server.transport == TransportMode::Quic
            && (self.server.cert_path.is_none() || self.server.key_path.is_none())
        {
            // QUIC with self-signed is OK, just warn
        }

        // Client validation
        for (i, ep) in self.client.servers.iter().enumerate() {
            if ep.address.is_empty() {
                errors.push(format!("client.servers[{}].address must not be empty", i));
            }
            if ep.token.is_empty() {
                errors.push(format!("client.servers[{}].token must not be empty", i));
            }
            // Validate address format (host:port)
            if !ep.address.is_empty() && !ep.address.contains(':') {
                errors.push(format!(
                    "client.servers[{}].address '{}' must be in host:port format",
                    i, ep.address
                ));
            }
        }

        // IPC listen address validation
        if let Some(ref addr) = self.client.ipc_listen_address {
            if !addr.is_empty() && !addr.contains(':') {
                errors.push(format!(
                    "client.ipc_listen_address '{}' must be in host:port format",
                    addr
                ));
            }
        }

        // Token validation
        for (i, tok) in self.security.tokens.iter().enumerate() {
            if tok.token.is_empty() {
                errors.push(format!("security.tokens[{}].token must not be empty", i));
            }
            if tok.name.is_empty() {
                errors.push(format!("security.tokens[{}].name must not be empty", i));
            }
            if tok.token.len() < 8 {
                errors.push(format!(
                    "security.tokens[{}].token should be at least 8 characters",
                    i
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Returns the default config file path based on platform conventions.
///
/// Search order:
/// 1. User config: `%APPDATA%\RGPU\rgpu.toml` (Windows) or `~/.config/rgpu/rgpu.toml` (Linux/macOS)
/// 2. System-wide config: `%PROGRAMDATA%\RGPU\rgpu.toml` (Windows) or `/etc/rgpu/rgpu.toml` (Linux/macOS)
/// 3. Local fallback: `./rgpu.toml`
pub fn default_config_path() -> String {
    #[cfg(windows)]
    {
        // Check user-local config first (always writable)
        if let Ok(appdata) = std::env::var("APPDATA") {
            let user_path = format!(r"{}\RGPU\rgpu.toml", appdata);
            if std::path::Path::new(&user_path).exists() {
                return user_path;
            }
        }
        // Fall back to system-wide config
        let programdata = std::env::var("PROGRAMDATA")
            .unwrap_or_else(|_| r"C:\ProgramData".to_string());
        let system_path = format!(r"{}\RGPU\rgpu.toml", programdata);
        if std::path::Path::new(&system_path).exists() {
            return system_path;
        }
    }
    #[cfg(not(windows))]
    {
        // Check user-local config first
        if let Some(home) = std::env::var_os("HOME") {
            let user_path = std::path::PathBuf::from(home)
                .join(".config/rgpu/rgpu.toml");
            if user_path.exists() {
                return user_path.to_string_lossy().to_string();
            }
        }
        let system_path = "/etc/rgpu/rgpu.toml";
        if std::path::Path::new(system_path).exists() {
            return system_path.to_string();
        }
    }
    "rgpu.toml".to_string()
}

/// Returns a writable config path, suitable for saving from the UI.
///
/// If the current config path is not writable, returns a user-local fallback path.
/// Also creates the parent directory if needed.
pub fn writable_config_path(current_path: &str) -> String {
    // Try writing to the current path first
    if is_path_writable(current_path) {
        return current_path.to_string();
    }

    // Fall back to user-local config directory
    let user_path = user_config_path();
    if let Some(parent) = std::path::Path::new(&user_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    user_path
}

/// Returns the user-local config path (always writable).
fn user_config_path() -> String {
    #[cfg(windows)]
    {
        let appdata = std::env::var("APPDATA")
            .unwrap_or_else(|_| r"C:\Users\Default\AppData\Roaming".to_string());
        format!(r"{}\RGPU\rgpu.toml", appdata)
    }
    #[cfg(not(windows))]
    {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        format!("{}/.config/rgpu/rgpu.toml", home)
    }
}

/// Check if a file path is writable (try opening for append).
fn is_path_writable(path: &str) -> bool {
    if std::path::Path::new(path).exists() {
        std::fs::OpenOptions::new()
            .write(true)
            .append(true)
            .open(path)
            .is_ok()
    } else {
        // File doesn't exist — check if parent dir is writable
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.exists() {
                return std::fs::create_dir_all(parent).is_ok();
            }
            // Try creating a temp file in the parent
            let test_path = parent.join(".rgpu_write_test");
            if std::fs::write(&test_path, b"").is_ok() {
                let _ = std::fs::remove_file(&test_path);
                true
            } else {
                false
            }
        } else {
            false
        }
    }
}

fn default_port() -> u16 {
    9876
}

fn default_bind() -> String {
    "0.0.0.0".to_string()
}

fn default_max_clients() -> u32 {
    16
}

fn default_true() -> bool {
    true
}
