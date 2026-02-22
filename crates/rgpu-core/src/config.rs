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
        Ok(config)
    }

    /// Load configuration from file if it exists, otherwise return defaults.
    pub fn load_or_default(path: &str) -> Self {
        Self::load(path).unwrap_or_default()
    }
}

/// Returns the default config file path based on platform conventions.
/// Search order:
/// 1. System-wide config: `%PROGRAMDATA%\RGPU\rgpu.toml` (Windows) or `/etc/rgpu/rgpu.toml` (Linux/macOS)
/// 2. Local fallback: `./rgpu.toml`
pub fn default_config_path() -> String {
    #[cfg(windows)]
    {
        let programdata = std::env::var("PROGRAMDATA")
            .unwrap_or_else(|_| r"C:\ProgramData".to_string());
        let system_path = format!(r"{}\RGPU\rgpu.toml", programdata);
        if std::path::Path::new(&system_path).exists() {
            return system_path;
        }
    }
    #[cfg(not(windows))]
    {
        let system_path = "/etc/rgpu/rgpu.toml";
        if std::path::Path::new(system_path).exists() {
            return system_path.to_string();
        }
    }
    "rgpu.toml".to_string()
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
