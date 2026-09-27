//! Anthropic API key storage: OS keychain / Secret Service, with a 0600 file
//! fallback for Linux desktops that run no Secret Service daemon.
//! `ANTHROPIC_API_KEY` in the environment overrides both.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const SERVICE: &str = "kz.kenes.app";
const ACCOUNT: &str = "anthropic-api-key";

pub struct Secrets {
    fallback_file: PathBuf,
}

impl Secrets {
    pub fn new(data_dir: &Path) -> Self {
        Self { fallback_file: data_dir.join("api_key") }
    }

    pub fn get(&self) -> Option<String> {
        if let Some(key) = std::env::var("ANTHROPIC_API_KEY").ok().filter(|k| !k.trim().is_empty()) {
            return Some(key);
        }
        match keyring::Entry::new(SERVICE, ACCOUNT).and_then(|e| e.get_password()) {
            Ok(key) => return Some(key),
            Err(keyring::Error::NoEntry) => {}
            Err(e) => log::warn!("keychain read failed, trying file fallback: {e}"),
        }
        std::fs::read_to_string(&self.fallback_file)
            .ok()
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
    }

    /// Stores the key; an empty key deletes it.
    pub fn set(&self, key: &str) -> Result<()> {
        let key = key.trim();
        let entry = keyring::Entry::new(SERVICE, ACCOUNT);
        if key.is_empty() {
            if let Ok(e) = entry {
                let _ = e.delete_credential();
            }
            let _ = std::fs::remove_file(&self.fallback_file);
            return Ok(());
        }
        match entry.and_then(|e| e.set_password(key)) {
            Ok(()) => {
                let _ = std::fs::remove_file(&self.fallback_file);
                Ok(())
            }
            Err(e) => {
                log::warn!("keychain write failed, storing key in a private file: {e}");
                write_private(&self.fallback_file, key)
            }
        }
    }
}

fn write_private(path: &Path, contents: &str) -> Result<()> {
    use std::io::Write;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(path).with_context(|| format!("writing {}", path.display()))?;
    f.write_all(contents.as_bytes())?;
    Ok(())
}
