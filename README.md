# foundry-local

Microsoft **Foundry Local** as a loadable RemoteMedia SDK cdylib
plugin. Runs OpenAI-shaped chat completions against locally-loaded
ONNX Runtime GenAI models — no cloud calls, no HTTP — by linking the
Foundry Local Core engine in-process via the official
[`foundry-local-sdk`](https://crates.io/crates/foundry-local-sdk)
Rust crate.

Path 3 of the SDK plugin spectrum (hand-rolled `AsyncStreamingNode`
via `remotemedia-plugin-sdk`). Lives as its own crate + repo because
`foundry-local-sdk`'s `build.rs` downloads hundreds of MB of native
libs (Foundry Core engine + ORT GenAI) from NuGet/ORT-Nightly — way
too heavy to drop into `remotemedia-core`'s default build.

## Nodes exposed

| `node_type` | Input → Output | Notes |
|---|---|---|
| `FoundryLocalChatNode` | `Text` / `Json` → streamed `Text` tokens | OpenAI-shaped chat completions, tool calling, per-session history. Reuses the same `ToolSpec` + `dispatch_tool_call` primitives as the in-tree `OpenAIChatNode`. |

Planned follow-ups (one file each, identical pattern):
- `FoundryLocalEmbeddingNode` — `EmbeddingClient` mapping `Text → Tensor`
- `FoundryLocalSTTNode` — `AudioClient.transcribe_streaming` mapping `Audio(16kHz mono F32) → Text`

## Manifest usage

```json
{
  "plugins": ["foundry-local@v0.1.0"],
  "nodes": [
    {
      "id": "chat",
      "node_type": "FoundryLocalChatNode",
      "params": {
        "app_name": "my-app",
        "model_alias": "phi-3.5-mini",
        "system_prompt": "You are a helpful assistant.",
        "temperature": 0.7,
        "max_tokens": 256,
        "enable_say_tool": true
      }
    }
  ],
  "connections": [
    { "from": "stt",  "to": "chat" },
    { "from": "chat", "to": "tts"  }
  ]
}
```

During development, replace the `plugins` entry with a local path:

```json
"plugins": ["./target/release/libfoundry_local_plugin.so"]
```

## Config reference

| Field (snake_case + camelCase alias) | Type | Default | Notes |
|---|---|---|---|
| `app_name` | `String` | `"remotemedia"` | Drives default app-data / cache / logs paths (`~/.{app_name}/...`). First-wins across nodes in one process. |
| `model_alias` | `String` | **required** | Foundry catalog alias (e.g. `"phi-3.5-mini"`). SDK auto-picks the best cached variant. |
| `model_cache_dir`, `app_data_dir`, `logs_dir`, `library_path` | `String` | SDK defaults | Override SDK defaults. |
| `log_level` | `"trace"…"fatal"` | `"warn"` | Foundry Core log level. |
| `system_prompt` | `String` | — | Prepended to every request. |
| `output_channel` | `String` | `"tts"` | Channel tag on emitted `RuntimeData::Text`. |
| `max_tokens`, `temperature`, `top_p`, `top_k`, `frequency_penalty`, `presence_penalty`, `n`, `random_seed` | numeric | SDK defaults | Sampling controls. `top_k` + `random_seed` are Foundry extensions. |
| `response_format` | `{type: "text" \| "json_object" \| "json_schema" \| "lark_grammar", ...}` | `text` | Output-format constraint. JSON-Schema / Lark-grammar variants take a `schema` / `grammar` string. |
| `history_turns` | `usize` | `10` | Conversation turns retained as context. `0` = stateless single-turn. |
| `streaming` | `bool` | `true` | Stream tokens vs. buffer-then-emit. |
| `prewarm` | `bool` | `false` | Download + load model during `initialize()` instead of on first request. |
| `register_execution_providers` | `bool` | `false` | Run `download_and_register_eps` at init for NPU / GPU EPs (Windows + WinML). |
| `enable_say_tool`, `enable_show_tool` | `bool` | `false` | Register the built-in `say` / `show` tools. |
| `tools` | `[ToolSpec]` | `[]` | User-defined tools (same shape as the in-tree `OpenAIChatNode`). |
| `active_tools` | `[String]` | `None` (= all) | Subset of registered tool names exposed to the model. |
| `tool_choice` | `"none" \| "auto" \| "required" \| {function: {name: "…"}}` | `None` | Tool-selection hint. |

## Engine sharing

The plugin holds one process-scoped `FoundryLocalManager` + a per-alias
`Arc<Model>` cache for the life of the dlopen'd `.so`. Two nodes in
the same manifest that reference the same alias share both the
downloaded weights and the loaded-into-VRAM handle. The provider
also coalesces concurrent first-loads via per-alias `OnceCell`s.

## Building

```sh
cargo build --release
# .so → ./target/release/libfoundry_local_plugin.so
```

First build pulls the Foundry Core + ORT GenAI native libs through
the SDK's `build.rs` (one-shot, cached in cargo's registry).
Subsequent builds skip the download.

For NPU / GPU acceleration on Windows, enable the `winml` SDK
feature in `Cargo.toml`:

```toml
foundry-local-sdk = { version = "1.1", features = ["winml"] }
```

## Why a separate repo

The `remotemedia-core` host workspace intentionally has no ML inference
deps after the recent extraction wave (`ort`, `ndarray`, `tokenizers`,
etc. all live in standalone plugin repos now — `silero-vad`,
`llama-cpp`, `lfm25-audio-onnx`, `whisper-to-vad`,
`speaker-diarization`, …). Foundry Local belongs in the same category:
every model run goes through the dlopen'd Foundry Core engine, so
isolating its native-lib download to one cdylib is a clean win.

This crate's `Cargo.toml` declares its own `[workspace]` so cargo
doesn't try to unify its deps with any consumer workspace.

## Shared LLM primitives

History bookkeeping (`HistoryEntry`, turn-bounded windowing), tool
schema (`ToolSpec`), the streaming `ToolCallAccum`, and
`dispatch_tool_call` come from the skinny
[`remotemedia-llm`](https://github.com/RemoteMedia-SDK/remotemedia-sdk/tree/main/crates/llm)
crate — the same source the in-tree `OpenAIChatNode` uses. Adding a
new built-in tool, fixing a history-window bug, or extending the tool
dispatch contract changes behavior in both places without code
duplication.

## License

See [LICENSE.md](./LICENSE.md) — RemoteMedia SDK Community License 1.0.
