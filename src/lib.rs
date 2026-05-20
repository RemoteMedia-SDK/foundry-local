//! Microsoft Foundry Local as a loadable cdylib plugin.
//!
//! Path 3 of the SDK plugin spectrum (hand-rolled
//! `AsyncStreamingNode` via `remotemedia-plugin-sdk`):
//!
//!   - **In-tree (Path 1)** would force every `cargo build` of core
//!     to pay `foundry-local-sdk`'s build.rs cost (NuGet + ORT-Nightly
//!     native-lib download, ~hundreds of MB).
//!   - **Dual-emit (Path 2)** would let the in-tree node also publish
//!     as a cdylib, but still drags the dep into the host workspace.
//!   - **Hand-rolled (this file)** keeps the SDK and its downloaded
//!     natives isolated to this one `.so` — the host workspace is
//!     untouched. Manifests opt in via
//!     `"plugins": ["foundry-local@v0.1.0"]`.
//!
//! ## What's exported
//!
//!   * **`FoundryLocalChatNode`** — OpenAI-shaped streaming chat
//!     completions. Reuses the same `HistoryEntry` / `ToolSpec` /
//!     `dispatch_tool_call` primitives the in-tree `OpenAIChatNode`
//!     uses, sourced from the skinny `remotemedia-llm` crate so
//!     there's one source of truth.
//!
//! ## What's NOT exported yet
//!
//!   * **`FoundryLocalEmbeddingNode`** — `EmbeddingClient.generate_embedding`
//!     mapping to `RuntimeData::Text → RuntimeData::Tensor`. Pattern is
//!     identical to `chat.rs`; left for a follow-up.
//!   * **`FoundryLocalSTTNode`** — `AudioClient.transcribe_streaming`.
//!     Needs a small spill-to-temp-WAV adapter since the SDK takes
//!     file paths; pattern is identical otherwise.
//!
//! ## Engine sharing
//!
//! [`provider`] holds a single `FoundryLocalManager` + a per-alias
//! `Arc<Model>` cache for the life of this dlopen'd library. Two
//! nodes in the same manifest that reference the same alias share
//! both the downloaded weights and the loaded-into-VRAM handle.

mod chat;
mod provider;

pub use chat::{
    FoundryLocalChatConfig, FoundryLocalChatNode, FoundryLocalChatNodeFactory, FoundryLogLevel,
    FoundryResponseFormat, FoundryToolChoice,
};

// ---------------------------------------------------------------------------
// Plugin registration
// ---------------------------------------------------------------------------

// One entry per exported factory. The host's
// `LoadableNodeBundle::load()` calls `list_factories()` (emitted by
// the macro) and registers each factory into the
// `StreamingNodeRegistry` under its `node_type()`.
remotemedia_plugin_sdk::plugin_export!(FoundryLocalChatNodeFactory);
