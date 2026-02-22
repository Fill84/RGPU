use clap::{Parser, Subcommand};
use tracing::info;

#[derive(Parser)]
#[command(name = "rgpu")]
#[command(about = "RGPU - Remote GPU sharing over the network")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the RGPU server (exposes local GPUs over the network)
    Server {
        /// Port to listen on
        #[arg(short, long, default_value_t = 9876)]
        port: u16,

        /// Bind address
        #[arg(short, long, default_value = "0.0.0.0")]
        bind: String,

        /// TLS certificate file (PEM)
        #[arg(long)]
        cert: Option<String>,

        /// TLS private key file (PEM)
        #[arg(long)]
        key: Option<String>,

        /// Configuration file path
        #[arg(short, long, default_value = "rgpu.toml")]
        config: String,

        /// Write PID to this file (for service managers)
        #[arg(long)]
        pid_file: Option<String>,
    },

    /// Start the RGPU client daemon (connects to servers and exposes remote GPUs locally)
    Client {
        /// Server address(es) to connect to (host:port)
        #[arg(short, long)]
        server: Vec<String>,

        /// Authentication token
        #[arg(short, long, default_value = "")]
        token: String,

        /// Configuration file path
        #[arg(short, long, default_value = "rgpu.toml")]
        config: String,

        /// Write PID to this file (for service managers)
        #[arg(long)]
        pid_file: Option<String>,
    },

    /// Generate an authentication token
    Token {
        /// Name for this token/client
        #[arg(short, long, default_value = "client")]
        name: String,
    },

    /// Show discovered GPU information (connects to a server and queries)
    Info {
        /// Server address to query
        #[arg(short, long)]
        server: String,

        /// Authentication token
        #[arg(short, long, default_value = "")]
        token: String,
    },

    /// Launch the RGPU desktop GUI
    Ui {
        /// Server address(es) to monitor (host:port)
        #[arg(short, long)]
        server: Vec<String>,

        /// Authentication token
        #[arg(short, long, default_value = "")]
        token: String,

        /// Configuration file path
        #[arg(short, long, default_value = "rgpu.toml")]
        config: String,

        /// Metrics poll interval in seconds
        #[arg(long, default_value_t = 2)]
        poll_interval: u64,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    rgpu_common::init_logging();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Server {
            port,
            bind,
            cert,
            key,
            config,
            pid_file,
        }) => {
            if let Some(ref path) = pid_file {
                std::fs::write(path, std::process::id().to_string())?;
            }

            info!("starting RGPU server on {}:{}", bind, port);

            let mut server_config = rgpu_core::config::ServerConfig::default();
            server_config.port = port;
            server_config.bind = bind;
            server_config.cert_path = cert;
            server_config.key_path = key;

            // Load tokens from config if available
            let rgpu_config = rgpu_core::config::RgpuConfig::load_or_default(&config);

            let server =
                rgpu_server::RgpuServer::new(server_config, rgpu_config.security.tokens);
            let result = server
                .run()
                .await
                .map_err(|e| anyhow::anyhow!("{}", e));

            if let Some(ref path) = pid_file {
                let _ = std::fs::remove_file(path);
            }

            result?;
        }

        Some(Commands::Client {
            server,
            token,
            config,
            pid_file,
        }) => {
            if let Some(ref path) = pid_file {
                std::fs::write(path, std::process::id().to_string())?;
            }

            info!("starting RGPU client daemon");

            let mut client_config = rgpu_core::config::ClientConfig::default();

            // Add servers from CLI args
            for addr in server {
                client_config.servers.push(rgpu_core::config::ServerEndpoint {
                    address: addr,
                    token: token.clone(),
                    ca_cert: None,
                    transport: rgpu_core::config::TransportMode::default(),
                });
            }

            // Merge with config file
            let rgpu_config = rgpu_core::config::RgpuConfig::load_or_default(&config);
            client_config.servers.extend(rgpu_config.client.servers);
            client_config.include_local_gpus = rgpu_config.client.include_local_gpus;
            client_config.gpu_ordering = rgpu_config.client.gpu_ordering;

            if client_config.servers.is_empty() && !client_config.include_local_gpus {
                anyhow::bail!("no servers configured and include_local_gpus is false. Use --server or add servers to rgpu.toml");
            }

            let daemon = rgpu_client::ClientDaemon::new(client_config);
            let result = daemon
                .run()
                .await
                .map_err(|e| anyhow::anyhow!("{}", e));

            if let Some(ref path) = pid_file {
                let _ = std::fs::remove_file(path);
            }

            result?;
        }

        Some(Commands::Token { name }) => {
            let token = rgpu_transport::auth::generate_token(32);
            println!("Generated RGPU token for '{}':", name);
            println!();
            println!("  {}", token);
            println!();
            println!("Add this to your server's rgpu.toml:");
            println!();
            println!("  [[security.tokens]]");
            println!("  token = \"{}\"", token);
            println!("  name = \"{}\"", name);
            println!();
            println!("And to your client's rgpu.toml or --token flag:");
            println!();
            println!("  [[client.servers]]");
            println!("  address = \"<server-ip>:9876\"");
            println!("  token = \"{}\"", token);
        }

        Some(Commands::Ui {
            server,
            token,
            config,
            poll_interval,
        }) => {
            info!("launching RGPU UI");

            // Collect servers from CLI args
            let mut all_servers: Vec<(String, String)> = server
                .into_iter()
                .map(|addr| (addr, token.clone()))
                .collect();

            // Also load servers from config file
            let rgpu_config = rgpu_core::config::RgpuConfig::load_or_default(&config);
            for endpoint in &rgpu_config.client.servers {
                all_servers.push((endpoint.address.clone(), endpoint.token.clone()));
            }

            rgpu_ui::launch_ui(all_servers, config, poll_interval)?;
        }

        Some(Commands::Info { server, token }) => {
            info!("querying GPU info from {}", server);

            use tokio::io::AsyncWriteExt;
            use tokio::net::TcpStream;

            let stream = TcpStream::connect(&server).await?;
            let (mut reader, mut writer) = stream.into_split();

            // Send Hello
            let hello = rgpu_protocol::messages::Message::Hello {
                protocol_version: rgpu_protocol::messages::PROTOCOL_VERSION,
                name: "RGPU Info".to_string(),
                challenge: None,
            };
            let frame = rgpu_protocol::wire::encode_message(&hello, 0)?;
            writer.write_all(&frame).await?;

            // Read server Hello
            let server_hello = read_message(&mut reader).await?;
            let challenge = match &server_hello {
                rgpu_protocol::messages::Message::Hello { challenge, .. } => {
                    challenge.clone().unwrap_or_default()
                }
                _ => Vec::new(),
            };

            // Authenticate
            let response =
                rgpu_transport::auth::compute_challenge_response(&token, &challenge);
            let auth_msg = rgpu_protocol::messages::Message::Authenticate {
                token: token.clone(),
                challenge_response: response,
            };
            let frame = rgpu_protocol::wire::encode_message(&auth_msg, 0)?;
            writer.write_all(&frame).await?;

            // Read auth result
            let auth_result = read_message(&mut reader).await?;

            match auth_result {
                rgpu_protocol::messages::Message::AuthResult {
                    success: true,
                    available_gpus,
                    ..
                } => {
                    println!("Connected to RGPU server at {}", server);
                    println!("Available GPUs:");
                    println!();
                    for (i, gpu) in available_gpus.iter().enumerate() {
                        println!("  GPU {}: {}", i, gpu.device_name);
                        println!("    Type:     {:?}", gpu.device_type);
                        println!(
                            "    VRAM:     {} MB",
                            gpu.total_memory / (1024 * 1024)
                        );
                        println!("    Vulkan:   {}", gpu.supports_vulkan);
                        println!("    CUDA:     {}", gpu.supports_cuda);
                        if let Some((maj, min)) = gpu.cuda_compute_capability {
                            println!("    Compute:  {}.{}", maj, min);
                        }
                        println!();
                    }
                }
                rgpu_protocol::messages::Message::AuthResult {
                    success: false,
                    error_message,
                    ..
                } => {
                    eprintln!(
                        "Authentication failed: {}",
                        error_message.unwrap_or_default()
                    );
                }
                _ => {
                    eprintln!("Unexpected response from server");
                }
            }
        }

        None => {
            // Default: launch UI when no subcommand is given (e.g. double-click on Windows/macOS)
            info!("launching RGPU UI (default)");
            let rgpu_config = rgpu_core::config::RgpuConfig::load_or_default("rgpu.toml");
            let servers: Vec<(String, String)> = rgpu_config
                .client
                .servers
                .iter()
                .map(|s| (s.address.clone(), s.token.clone()))
                .collect();
            rgpu_ui::launch_ui(servers, "rgpu.toml".to_string(), 2)?;
        }
    }

    Ok(())
}

async fn read_message<R: tokio::io::AsyncRead + Unpin>(
    reader: &mut R,
) -> anyhow::Result<rgpu_protocol::messages::Message> {
    use tokio::io::AsyncReadExt;

    let mut header_buf = [0u8; rgpu_protocol::wire::HEADER_SIZE];
    reader.read_exact(&mut header_buf).await?;
    let (flags, _, payload_len) = rgpu_protocol::wire::decode_header(&header_buf)?;
    let mut payload = vec![0u8; payload_len as usize];
    reader.read_exact(&mut payload).await?;
    let msg = rgpu_protocol::wire::decode_message(&payload, flags)?;
    Ok(msg)
}
