//! QUIC transport for RGPU, using Quinn.
//!
//! QUIC always provides encryption (TLS 1.3), so there is no "plain" mode.
//! Each request-response pair uses a bidirectional QUIC stream.

use std::net::SocketAddr;
use std::sync::Arc;

use quinn::{Endpoint, ServerConfig as QuinnServerConfig};
use tracing::{debug, error, info};

use rgpu_protocol::messages::Message;
use rgpu_protocol::wire;

use crate::error::TransportError;

/// Build a QUIC server endpoint.
pub fn build_quic_server(
    bind_addr: SocketAddr,
    cert_path: &str,
    key_path: &str,
) -> Result<Endpoint, TransportError> {
    let (certs, key) = crate::tls::load_certs_and_key(cert_path, key_path)
        .map_err(|e| TransportError::Quic(format!("failed to load certs: {}", e)))?;

    let mut server_crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| TransportError::Tls(e))?;

    server_crypto.alpn_protocols = vec![b"rgpu/1".to_vec()];

    let mut server_config = QuinnServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)
            .map_err(|e| TransportError::Quic(e.to_string()))?,
    ));

    // Set idle timeout for dead connection detection
    let mut transport_config = quinn::TransportConfig::default();
    transport_config.max_idle_timeout(Some(
        std::time::Duration::from_secs(120)
            .try_into()
            .expect("valid idle timeout"),
    ));
    server_config.transport_config(Arc::new(transport_config));

    let endpoint = Endpoint::server(server_config, bind_addr)
        .map_err(|e| TransportError::Quic(format!("failed to bind QUIC endpoint: {}", e)))?;

    info!("QUIC server listening on {}", bind_addr);
    Ok(endpoint)
}

/// Opaque wrapper around a QUIC connection.
/// Allows the client daemon to hold a QUIC connection without depending on quinn directly.
pub struct QuicConnection {
    connection: quinn::Connection,
}

impl QuicConnection {
    /// Send a message and receive a response over this QUIC connection.
    /// Opens a new bidirectional stream for each request.
    pub async fn send_and_receive(
        &self,
        msg: &Message,
    ) -> Result<Message, TransportError> {
        quic_send_and_receive(&self.connection, msg).await
    }

    /// Get the remote address of this connection.
    pub fn remote_address(&self) -> SocketAddr {
        self.connection.remote_address()
    }
}

/// Build a QUIC client endpoint and connect to the server.
pub async fn connect_quic_client(
    server_addr: &str,
) -> Result<QuicConnection, TransportError> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    let mut client_crypto = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();

    client_crypto.alpn_protocols = vec![b"rgpu/1".to_vec()];
    // Allow self-signed certs for development
    client_crypto.dangerous().set_certificate_verifier(Arc::new(SkipServerVerification));

    let client_config = quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(client_crypto)
            .map_err(|e| TransportError::Quic(e.to_string()))?,
    ));

    let bind_addr: SocketAddr = "0.0.0.0:0".parse()
        .map_err(|e| TransportError::Quic(format!("invalid bind address: {}", e)))?;
    let mut endpoint = Endpoint::client(bind_addr)
        .map_err(|e| TransportError::Quic(format!("failed to create QUIC client endpoint: {}", e)))?;
    endpoint.set_default_client_config(client_config);

    let addr: SocketAddr = server_addr
        .parse()
        .map_err(|e| TransportError::Quic(format!("invalid server address: {}", e)))?;

    let connection = endpoint
        .connect(addr, "rgpu-server")
        .map_err(|e| TransportError::Quic(format!("QUIC connect error: {}", e)))?
        .await
        .map_err(|e| TransportError::Quic(format!("QUIC connection error: {}", e)))?;

    debug!("QUIC connection established to {}", server_addr);
    Ok(QuicConnection { connection })
}

/// Send a message and receive a response over a QUIC connection.
/// Each call opens a new bidirectional stream.
pub async fn quic_send_and_receive(
    connection: &quinn::Connection,
    msg: &Message,
) -> Result<Message, TransportError> {
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .map_err(|e| TransportError::Quic(format!("open stream error: {}", e)))?;

    // Encode and send the message
    let frame = wire::encode_message(msg, 0)
        .map_err(|e| TransportError::Wire(e))?;

    send.write_all(&frame)
        .await
        .map_err(|e| TransportError::Quic(format!("write error: {}", e)))?;

    send.finish()
        .map_err(|e| TransportError::Quic(format!("finish error: {}", e)))?;

    // Read response
    let response = read_quic_message(&mut recv).await?;
    Ok(response)
}

/// Read a single framed message from a QUIC receive stream.
pub async fn read_quic_message(
    recv: &mut quinn::RecvStream,
) -> Result<Message, TransportError> {
    let mut header_buf = [0u8; wire::HEADER_SIZE];
    recv.read_exact(&mut header_buf)
        .await
        .map_err(|e| TransportError::Quic(format!("read header error: {}", e)))?;

    let (flags, _, payload_len) = wire::decode_header(&header_buf)?;

    let mut payload = vec![0u8; payload_len as usize];
    recv.read_exact(&mut payload)
        .await
        .map_err(|e| TransportError::Quic(format!("read payload error: {}", e)))?;

    let msg = wire::decode_message(&payload, flags)?;
    Ok(msg)
}

/// Handle an incoming QUIC bidirectional stream (server side).
/// Reads one request, calls the handler, and writes the response.
pub async fn handle_quic_stream(
    send: &mut quinn::SendStream,
    recv: &mut quinn::RecvStream,
    handler: &(dyn Fn(Message) -> Option<Message> + Send + Sync),
) -> Result<(), TransportError> {
    let msg = read_quic_message(recv).await?;

    if let Some(response) = handler(msg) {
        let frame = wire::encode_message(&response, 0)
            .map_err(|e| TransportError::Wire(e))?;

        send.write_all(&frame)
            .await
            .map_err(|e| TransportError::Quic(format!("write response error: {}", e)))?;

        send.finish()
            .map_err(|e| TransportError::Quic(format!("finish response error: {}", e)))?;
    }

    Ok(())
}

/// Accept and handle QUIC connections in a loop (server side).
pub async fn accept_quic_connections(
    endpoint: Endpoint,
    handler: Arc<dyn Fn(Message) -> Option<Message> + Send + Sync>,
) {
    while let Some(incoming) = endpoint.accept().await {
        let handler = handler.clone();

        tokio::spawn(async move {
            match incoming.await {
                Ok(connection) => {
                    let remote = connection.remote_address();
                    info!("QUIC client connected from {}", remote);

                    loop {
                        match connection.accept_bi().await {
                            Ok((mut send, mut recv)) => {
                                let handler = handler.clone();
                                tokio::spawn(async move {
                                    if let Err(e) = handle_quic_stream(&mut send, &mut recv, &*handler).await {
                                        debug!("QUIC stream error: {}", e);
                                    }
                                });
                            }
                            Err(quinn::ConnectionError::ApplicationClosed(_)) => {
                                debug!("QUIC client {} disconnected", remote);
                                break;
                            }
                            Err(e) => {
                                error!("QUIC accept stream error from {}: {}", remote, e);
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("QUIC incoming connection error: {}", e);
                }
            }
        });
    }
}

/// Certificate verifier that accepts any server certificate (development only).
#[derive(Debug)]
struct SkipServerVerification;

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::RSA_PKCS1_SHA384,
            rustls::SignatureScheme::RSA_PKCS1_SHA512,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ECDSA_NISTP521_SHA512,
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::ED448,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
        ]
    }
}
