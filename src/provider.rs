//! Process-scoped Foundry Local engine + model handle cache.
//!
//! One [`FoundryLocalManager`] per dlopen'd `.so` (the SDK loads
//! native libs from `OUT_DIR` and starts the Core engine on
//! construction — expensive to repeat). One `Arc<Model>` per loaded
//! alias so two pipelines that reference `phi-3.5-mini` share both
//! the downloaded weights AND the loaded-into-VRAM handle.
//!
//! The provider is owned by a `static OnceLock<Arc<Provider>>` inside
//! this `.so`. Each cdylib gets its own copy (dlopen'd libs don't
//! share statics with the host), but within one library load every
//! Foundry-backed node calls into the same engine.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use foundry_local_sdk::{FoundryLocalConfig, FoundryLocalManager, LogLevel, Model};
use parking_lot::Mutex;
use remotemedia_plugin_sdk::types::Error;
use tokio::sync::OnceCell;

/// Process-scoped Foundry engine + model cache.
pub struct Provider {
    /// Lazy engine handle. `FoundryLocalManager::create` returns a
    /// `&'static FoundryLocalManager` (the SDK owns the storage in
    /// a process-singleton OnceLock), so we just cache the reference.
    /// Loads native libraries + starts Core on first call.
    manager: OnceCell<&'static FoundryLocalManager>,
    /// `app_name` snapshot at first init. Subsequent provider
    /// requests that pass a different `app_name` are honoured by
    /// surfacing a warning; the first wins, because changing
    /// `app_name` mid-process would orphan the running engine.
    app_name: Mutex<Option<String>>,
    /// Per-alias model cache. Each entry is an `Arc<OnceCell<…>>` so
    /// concurrent first-load requests for the same alias coalesce.
    models: Mutex<HashMap<String, Arc<OnceCell<Arc<Model>>>>>,
    /// Tracks whether we already registered execution providers — we
    /// only want that to fire once per process even if multiple
    /// factories request it.
    eps_registered: OnceCell<()>,
}

impl Provider {
    fn new() -> Self {
        Self {
            manager: OnceCell::new(),
            app_name: Mutex::new(None),
            models: Mutex::new(HashMap::new()),
            eps_registered: OnceCell::new(),
        }
    }

    /// Build the SDK config from the per-node config snapshot. Only
    /// honoured on first call (which actually instantiates the
    /// engine); subsequent calls return the existing engine.
    fn build_sdk_config(snap: &ProviderConfigSnapshot) -> FoundryLocalConfig {
        let mut cfg = FoundryLocalConfig::new(&snap.app_name);
        if let Some(ref dir) = snap.app_data_dir {
            cfg = cfg.app_data_dir(dir);
        }
        if let Some(ref dir) = snap.model_cache_dir {
            cfg = cfg.model_cache_dir(dir);
        }
        if let Some(ref dir) = snap.logs_dir {
            cfg = cfg.logs_dir(dir);
        }
        if let Some(level) = snap.log_level {
            cfg = cfg.log_level(level);
        }
        if let Some(ref path) = snap.library_path {
            cfg = cfg.library_path(path);
        }
        cfg
    }

    /// Lazily construct the shared `FoundryLocalManager`. Concurrent
    /// callers all observe the same `&'static` engine handle.
    pub async fn ensure_manager(
        &self,
        snap: &ProviderConfigSnapshot,
    ) -> Result<&'static FoundryLocalManager, Error> {
        // Warn (but don't error) if a later node asks for a different
        // app_name — we honour first-wins.
        {
            let mut guard = self.app_name.lock();
            match guard.as_deref() {
                None => *guard = Some(snap.app_name.clone()),
                Some(existing) if existing != snap.app_name => {
                    tracing::warn!(
                        existing = %existing,
                        requested = %snap.app_name,
                        "[foundry-local] ignoring `app_name` override; provider already initialised with a different name"
                    );
                }
                _ => {}
            }
        }
        let mgr: &'static FoundryLocalManager = *self
            .manager
            .get_or_try_init(|| async {
                let cfg = Self::build_sdk_config(snap);
                let manager = FoundryLocalManager::create(cfg).map_err(|e| {
                    Error::Execution(format!(
                        "[foundry-local] FoundryLocalManager::create failed: {}",
                        e
                    ))
                })?;
                Ok::<_, Error>(manager)
            })
            .await?;
        if snap.register_execution_providers {
            self.eps_registered
                .get_or_try_init(|| async move {
                    mgr.download_and_register_eps(None).await.map_err(|e| {
                        Error::Execution(format!(
                            "[foundry-local] download_and_register_eps failed: {}",
                            e
                        ))
                    })?;
                    Ok::<_, Error>(())
                })
                .await?;
        }
        Ok(mgr)
    }

    /// Lazily download + load a model by alias. Multiple nodes
    /// referencing the same alias share one `Arc<Model>`.
    pub async fn ensure_model(
        &self,
        snap: &ProviderConfigSnapshot,
        alias: &str,
    ) -> Result<Arc<Model>, Error> {
        // Phase 1: get-or-create the per-alias `OnceCell` slot. We do
        // this under a quick `Mutex` so that the second concurrent
        // caller observes the same `OnceCell` the first one
        // installed.
        let slot = {
            let mut models = self.models.lock();
            models
                .entry(alias.to_string())
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };
        // Phase 2: drive the OnceCell — only the winner pays the
        // load cost; everyone else awaits the same future.
        let alias_owned = alias.to_string();
        let provider_self = self;
        let snap_clone = snap.clone();
        let model = slot
            .get_or_try_init(|| async move {
                let manager = provider_self.ensure_manager(&snap_clone).await?;
                // `Catalog::get_model` already returns `Arc<Model>` —
                // don't double-wrap; just hand the Arc straight back.
                let model = manager
                    .catalog()
                    .get_model(&alias_owned)
                    .await
                    .map_err(|e| {
                        Error::Execution(format!(
                            "[foundry-local] catalog lookup for `{}` failed: {}",
                            alias_owned, e
                        ))
                    })?;
                model.load().await.map_err(|e| {
                    Error::Execution(format!(
                        "[foundry-local] model `{}` failed to load: {}",
                        alias_owned, e
                    ))
                })?;
                Ok::<_, Error>(model)
            })
            .await?;
        Ok(model.clone())
    }
}

/// The plugin-local provider singleton.
///
/// One per loaded `.so` — dlopen'd libraries do not share statics
/// with the host. Within one library load every Foundry node calls
/// through here.
static PROVIDER: OnceLock<Arc<Provider>> = OnceLock::new();

pub fn provider() -> Arc<Provider> {
    PROVIDER.get_or_init(|| Arc::new(Provider::new())).clone()
}

/// Subset of node config that drives Foundry engine construction.
/// Snapshotted at node-create time so the async ensure path doesn't
/// have to hold a node-config borrow across `.await`.
#[derive(Debug, Clone)]
pub struct ProviderConfigSnapshot {
    pub app_name: String,
    pub app_data_dir: Option<String>,
    pub model_cache_dir: Option<String>,
    pub logs_dir: Option<String>,
    pub log_level: Option<LogLevel>,
    pub library_path: Option<String>,
    pub register_execution_providers: bool,
}
