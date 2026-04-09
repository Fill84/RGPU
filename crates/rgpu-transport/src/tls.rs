use std::sync::Arc;

use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use tokio_rustls::{TlsAcceptor, TlsConnector};

/// Load certificate chain and private key from PEM files.
pub fn load_certs_and_key(
    cert_path: &str,
    key_path: &str,
) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>), Box<dyn std::error::Error + Send + Sync>> {
    let cert_pem = std::fs::read(cert_path)?;
    let key_pem = std::fs::read(key_path)?;

    let certs: Vec<CertificateDer<'static>> = rustls_pemfile::certs(&mut &cert_pem[..])
        .collect::<Result<Vec<_>, _>>()?;

    let key: PrivateKeyDer<'static> = rustls_pemfile::private_key(&mut &key_pem[..])?
        .ok_or("no private key found in key file")?;

    Ok((certs, key))
}

/// Build a TLS acceptor for the server side.
pub fn build_server_tls(
    cert_path: &str,
    key_path: &str,
) -> Result<TlsAcceptor, Box<dyn std::error::Error + Send + Sync>> {
    let cert_pem = std::fs::read(cert_path)?;
    let key_pem = std::fs::read(key_path)?;

    let certs: Vec<CertificateDer<'static>> = rustls_pemfile::certs(&mut &cert_pem[..])
        .collect::<Result<Vec<_>, _>>()?;

    let key: PrivateKeyDer<'static> = rustls_pemfile::private_key(&mut &key_pem[..])?
        .ok_or("no private key found in key file")?;

    let config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)?;

    Ok(TlsAcceptor::from(Arc::new(config)))
}

/// Build a TLS connector for the client side.
pub fn build_client_tls(
    ca_cert_path: Option<&str>,
) -> Result<TlsConnector, Box<dyn std::error::Error + Send + Sync>> {
    let mut root_store = rustls::RootCertStore::empty();

    if let Some(ca_path) = ca_cert_path {
        let ca_pem = std::fs::read(ca_path)?;
        let certs: Vec<CertificateDer<'static>> = rustls_pemfile::certs(&mut &ca_pem[..])
            .collect::<Result<Vec<_>, _>>()?;
        for cert in certs {
            root_store.add(cert)?;
        }
    } else {
        root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    }

    let config = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();

    Ok(TlsConnector::from(Arc::new(config)))
}

/// Build a TLS connector that accepts any certificate.
///
/// **WARNING**: This disables all certificate verification and is only available
/// in debug builds. Using this in production exposes connections to MITM attacks.
#[cfg(debug_assertions)]
pub fn build_insecure_client_tls() -> Result<TlsConnector, Box<dyn std::error::Error + Send + Sync>>
{
    tracing::warn!("using insecure TLS - certificate verification disabled (development only)");
    let config = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(InsecureVerifier))
        .with_no_client_auth();

    Ok(TlsConnector::from(Arc::new(config)))
}

/// Stub for release builds — insecure TLS is not available.
#[cfg(not(debug_assertions))]
pub fn build_insecure_client_tls() -> Result<TlsConnector, Box<dyn std::error::Error + Send + Sync>>
{
    Err("insecure TLS is not available in release builds".into())
}

/// Certificate verifier that accepts everything (DEVELOPMENT ONLY).
#[cfg(debug_assertions)]
#[derive(Debug)]
struct InsecureVerifier;

#[cfg(debug_assertions)]
impl rustls::client::danger::ServerCertVerifier for InsecureVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::aws_lc_rs::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}
