use std::path::PathBuf;

/// The role this RGPU installation is configured for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Server,
    Client,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::Server => "server",
            Role::Client => "client",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "server" => Some(Role::Server),
            "client" => Some(Role::Client),
            _ => None,
        }
    }
}

/// Path to the role file.
fn role_file_path() -> PathBuf {
    #[cfg(windows)]
    {
        let base = std::env::var("ProgramData")
            .unwrap_or_else(|_| r"C:\ProgramData".to_string());
        PathBuf::from(base).join("RGPU").join("role")
    }
    #[cfg(not(windows))]
    {
        PathBuf::from("/etc/rgpu/role")
    }
}

/// Read the configured role from the role file.
/// Returns None if no role file exists (unconfigured installation).
pub fn get_installed_role() -> Option<Role> {
    let path = role_file_path();
    match std::fs::read_to_string(&path) {
        Ok(contents) => Role::from_str(&contents),
        Err(_) => None,
    }
}

/// Write the role file. Requires admin/root on first write.
pub fn set_role(role: Role) -> Result<(), String> {
    let path = role_file_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {}", parent.display(), e))?;
    }
    std::fs::write(&path, role.as_str())
        .map_err(|e| format!("failed to write role file {}: {}", path.display(), e))
}

/// Check if the given command is allowed for the installed role.
/// Returns Ok(()) if allowed, Err with message if not.
pub fn check_role(command_role: Role) -> Result<(), String> {
    match get_installed_role() {
        None => {
            // No role file — first run, auto-configure
            set_role(command_role)?;
            Ok(())
        }
        Some(installed) if installed == command_role => Ok(()),
        Some(installed) => Err(format!(
            "this installation is configured as '{}' — cannot run '{}' command.\n\
             To change role, run: rgpu set-role {}",
            installed.as_str(),
            command_role.as_str(),
            command_role.as_str(),
        )),
    }
}
