/// A system-wide instance lock that prevents double execution.
/// The lock is held for the lifetime of this struct.
/// Drop releases the lock automatically.
pub struct InstanceLock {
    #[cfg(unix)]
    _file: std::fs::File,
    #[cfg(windows)]
    _handle: windows_sys::Win32::Foundation::HANDLE,
}

impl InstanceLock {
    /// Try to acquire an instance lock for the given role.
    /// Returns Ok(lock) if acquired, Err if another instance is running.
    pub fn try_acquire(role: &str) -> Result<Self, String> {
        #[cfg(unix)]
        {
            Self::try_acquire_unix(role)
        }
        #[cfg(windows)]
        {
            Self::try_acquire_windows(role)
        }
    }

    #[cfg(unix)]
    fn try_acquire_unix(role: &str) -> Result<Self, String> {
        use std::fs::File;
        use std::os::unix::io::AsRawFd;

        let lock_path = format!("/run/rgpu-{}.lock", role);
        let file = File::create(&lock_path)
            .map_err(|e| format!("failed to create lock file {}: {}", lock_path, e))?;

        let ret = unsafe {
            libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB)
        };

        if ret != 0 {
            return Err(format!(
                "another RGPU {} is already running (lock: {})",
                role, lock_path
            ));
        }

        // Write PID for debugging
        use std::io::Write;
        let mut f = &file;
        let _ = writeln!(f, "{}", std::process::id());

        Ok(Self { _file: file })
    }

    #[cfg(windows)]
    fn try_acquire_windows(role: &str) -> Result<Self, String> {
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;

        let mutex_name: Vec<u16> = OsStr::new(&format!("Global\\RGPU_{}", role))
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        let handle = unsafe {
            windows_sys::Win32::System::Threading::CreateMutexW(
                std::ptr::null(),
                1, // bInitialOwner = TRUE
                mutex_name.as_ptr(),
            )
        };

        if handle == std::ptr::null_mut() {
            return Err(format!("failed to create mutex for RGPU {}", role));
        }

        let last_error = unsafe {
            windows_sys::Win32::Foundation::GetLastError()
        };

        // ERROR_ALREADY_EXISTS = 183
        if last_error == 183 {
            unsafe {
                windows_sys::Win32::Foundation::CloseHandle(handle);
            }
            return Err(format!(
                "another RGPU {} is already running",
                role
            ));
        }

        Ok(Self { _handle: handle })
    }
}

#[cfg(windows)]
impl Drop for InstanceLock {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::System::Threading::ReleaseMutex(self._handle);
            windows_sys::Win32::Foundation::CloseHandle(self._handle);
        }
    }
}
