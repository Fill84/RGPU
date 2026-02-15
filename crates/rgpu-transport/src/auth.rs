use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Compute HMAC-SHA256 challenge response.
/// The client computes HMAC(token, challenge) to prove possession of the token
/// without sending it in plaintext (though TLS already encrypts the channel).
pub fn compute_challenge_response(token: &str, challenge: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(token.as_bytes())
        .expect("HMAC can take key of any size");
    mac.update(challenge);
    mac.finalize().into_bytes().to_vec()
}

/// Verify a challenge response against the expected token.
pub fn verify_challenge_response(
    token: &str,
    challenge: &[u8],
    response: &[u8],
) -> bool {
    let mut mac = HmacSha256::new_from_slice(token.as_bytes())
        .expect("HMAC can take key of any size");
    mac.update(challenge);
    mac.verify_slice(response).is_ok()
}

/// Generate a random challenge of the given length.
pub fn generate_challenge(len: usize) -> Vec<u8> {
    use rand::RngCore;
    let mut challenge = vec![0u8; len];
    rand::thread_rng().fill_bytes(&mut challenge);
    challenge
}

/// Generate a random token (hex-encoded) for configuration.
pub fn generate_token(bytes: usize) -> String {
    use rand::RngCore;
    let mut token_bytes = vec![0u8; bytes];
    rand::thread_rng().fill_bytes(&mut token_bytes);
    hex::encode(token_bytes)
}
