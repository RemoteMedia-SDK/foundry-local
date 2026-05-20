//! `FoundryLocalChatNode` — OpenAI-shaped chat completions backed by
//! the in-process Foundry Local engine.
//!
//! Re-uses [`remotemedia_llm`] for `HistoryEntry`, `ToolSpec`,
//! `ToolCallAccum`, and `dispatch_tool_call` — same source of truth
//! the in-tree `OpenAIChatNode` reads. The transport, however, is
//! the Foundry Local Rust SDK (`foundry-local-sdk`), not HTTP/SSE.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use foundry_local_sdk::openai::ChatClient;
use foundry_local_sdk::{
    ChatCompletionRequestMessage, ChatCompletionTools, ChatResponseFormat, ChatToolChoice,
    LogLevel, Model,
};
use parking_lot::Mutex;
use remotemedia_llm::{
    dispatch_tool_call, to_openai_tools_array, window_start, HistoryEntry, ToolCallAccum, ToolSpec,
    default_say_tool, default_show_tool,
};
use remotemedia_plugin_sdk::abi_stable::sabi_trait::TD_Opaque;
use remotemedia_plugin_sdk::abi_stable::std_types::{RErr, ROk, RResult, RString};
use remotemedia_plugin_sdk::adapter::StreamingNodeFfiAdapter;
use remotemedia_plugin_sdk::traits::runtime_context::InitializeContextRead;
use remotemedia_plugin_sdk::traits::streaming::AsyncStreamingNode;
use remotemedia_plugin_sdk::types::{tag_text_str, Error, RuntimeData, TEXT_CHANNEL_DEFAULT};
use remotemedia_plugin_sdk::{FfiNodeBox, FfiNodeFactory, FfiNode_TO};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::OnceCell;
use tokio_stream::StreamExt;

use crate::provider::{provider, ProviderConfigSnapshot};

// ---------------------------------------------------------------------------
// Config — serde-friendly mirrors of the SDK enums so manifests can
// declare them inline.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FoundryResponseFormat {
    Text,
    JsonObject,
    JsonSchema { schema: String },
    LarkGrammar { grammar: String },
}

impl Default for FoundryResponseFormat {
    fn default() -> Self {
        Self::Text
    }
}

impl FoundryResponseFormat {
    fn into_sdk(self) -> ChatResponseFormat {
        match self {
            Self::Text => ChatResponseFormat::Text,
            Self::JsonObject => ChatResponseFormat::JsonObject,
            Self::JsonSchema { schema } => ChatResponseFormat::JsonSchema(schema),
            Self::LarkGrammar { grammar } => ChatResponseFormat::LarkGrammar(grammar),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FoundryToolChoice {
    None,
    Auto,
    Required,
    /// Force the model to call a specific function by name. The SDK's
    /// underlying `ChatToolChoice::Function` is a `String`-tuple
    /// variant; we expose a struct variant so the manifest shape
    /// `{"function": {"name": "say"}}` reads cleanly.
    Function { name: String },
}

impl FoundryToolChoice {
    fn into_sdk(self) -> ChatToolChoice {
        match self {
            Self::None => ChatToolChoice::None,
            Self::Auto => ChatToolChoice::Auto,
            Self::Required => ChatToolChoice::Required,
            Self::Function { name } => ChatToolChoice::Function(name),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FoundryLogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
}

impl FoundryLogLevel {
    fn into_sdk(self) -> LogLevel {
        match self {
            Self::Trace => LogLevel::Trace,
            Self::Debug => LogLevel::Debug,
            Self::Info => LogLevel::Info,
            Self::Warn => LogLevel::Warn,
            Self::Error => LogLevel::Error,
            Self::Fatal => LogLevel::Fatal,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct FoundryLocalChatConfig {
    /// App name passed to `FoundryLocalConfig::new`. Drives default
    /// app-data / cache / logs paths (`~/.{app_name}/...`). The first
    /// node to initialise the provider wins — subsequent nodes that
    /// pass a different `app_name` get a warning and keep using the
    /// original.
    #[serde(alias = "appName")]
    pub app_name: String,

    /// Foundry catalog alias of the model to load (e.g.
    /// `"phi-3.5-mini"`). The SDK auto-picks the best cached variant.
    #[serde(alias = "modelAlias")]
    pub model_alias: Option<String>,

    #[serde(alias = "modelCacheDir")]
    pub model_cache_dir: Option<String>,
    #[serde(alias = "appDataDir")]
    pub app_data_dir: Option<String>,
    #[serde(alias = "logsDir")]
    pub logs_dir: Option<String>,
    #[serde(alias = "logLevel")]
    pub log_level: Option<FoundryLogLevel>,
    #[serde(alias = "libraryPath")]
    pub library_path: Option<String>,

    #[serde(alias = "systemPrompt")]
    pub system_prompt: Option<String>,

    /// Output channel tag for emitted text tokens. Default: `"tts"`.
    #[serde(alias = "outputChannel")]
    pub output_channel: Option<String>,

    #[serde(alias = "maxTokens")]
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    #[serde(alias = "topP")]
    pub top_p: Option<f32>,
    #[serde(alias = "topK")]
    pub top_k: Option<u32>,
    #[serde(alias = "frequencyPenalty")]
    pub frequency_penalty: Option<f32>,
    #[serde(alias = "presencePenalty")]
    pub presence_penalty: Option<f32>,
    pub n: Option<u32>,
    #[serde(alias = "randomSeed")]
    pub random_seed: Option<u64>,

    #[serde(alias = "responseFormat")]
    pub response_format: Option<FoundryResponseFormat>,

    /// Conversation turns retained as context. Default `10`. Set to
    /// `0` for stateless single-turn mode.
    #[serde(alias = "historyTurns")]
    pub history_turns: usize,

    /// Streaming on by default. When `false`, the full response is
    /// buffered then emitted as one chunk.
    pub streaming: bool,

    /// Pay download + load cost during `initialize()` instead of on
    /// the first request. Useful with a warm session pool.
    pub prewarm: bool,

    /// Run `download_and_register_eps` at engine init for NPU/GPU EPs
    /// (Windows + `winml` feature). Default `false`.
    #[serde(alias = "registerExecutionProviders")]
    pub register_execution_providers: bool,

    #[serde(alias = "enableSayTool")]
    pub enable_say_tool: bool,
    #[serde(alias = "enableShowTool")]
    pub enable_show_tool: bool,
    #[serde(default, alias = "tools")]
    pub tools: Vec<ToolSpec>,
    #[serde(default, alias = "activeTools")]
    pub active_tools: Option<Vec<String>>,
    #[serde(default, alias = "toolChoice")]
    pub tool_choice: Option<FoundryToolChoice>,
}

impl Default for FoundryLocalChatConfig {
    fn default() -> Self {
        Self {
            app_name: "remotemedia".to_string(),
            model_alias: None,
            model_cache_dir: None,
            app_data_dir: None,
            logs_dir: None,
            log_level: None,
            library_path: None,
            system_prompt: None,
            output_channel: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            n: None,
            random_seed: None,
            response_format: None,
            history_turns: 10,
            streaming: true,
            prewarm: false,
            register_execution_providers: false,
            enable_say_tool: false,
            enable_show_tool: false,
            tools: Vec::new(),
            active_tools: None,
            tool_choice: None,
        }
    }
}

impl FoundryLocalChatConfig {
    fn provider_snapshot(&self) -> ProviderConfigSnapshot {
        ProviderConfigSnapshot {
            app_name: self.app_name.clone(),
            app_data_dir: self.app_data_dir.clone(),
            model_cache_dir: self.model_cache_dir.clone(),
            logs_dir: self.logs_dir.clone(),
            log_level: self.log_level.map(FoundryLogLevel::into_sdk),
            library_path: self.library_path.clone(),
            register_execution_providers: self.register_execution_providers,
        }
    }

    fn resolve_model_alias(&self) -> Result<&str, Error> {
        self.model_alias.as_deref().ok_or_else(|| {
            Error::InvalidData(
                "FoundryLocalChatNode requires `model_alias` in config (e.g. \"phi-3.5-mini\")"
                    .to_string(),
            )
        })
    }

    fn build_tool_registry(&self) -> Vec<ToolSpec> {
        let mut out: Vec<ToolSpec> = Vec::new();
        if self.enable_say_tool {
            out.push(default_say_tool());
        }
        if self.enable_show_tool {
            out.push(default_show_tool());
        }
        for spec in &self.tools {
            if let Some(existing) = out.iter_mut().find(|t| t.name == spec.name) {
                *existing = spec.clone();
            } else {
                out.push(spec.clone());
            }
        }
        if let Some(ref active) = self.active_tools {
            let active: HashSet<&str> = active.iter().map(String::as_str).collect();
            out.retain(|t| active.contains(t.name.as_str()));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

pub struct FoundryLocalChatNode {
    config: FoundryLocalChatConfig,
    /// Cached `Arc<Model>` after first load. The provider holds the
    /// authoritative cache; this slot is a per-node fast-path.
    model: OnceCell<Arc<Model>>,
    /// Per-session conversation history. Same shape as the in-tree
    /// `OpenAIChatNode` — `HistoryEntry` is the shared type.
    history: Arc<Mutex<HashMap<String, Vec<HistoryEntry>>>>,
}

impl FoundryLocalChatNode {
    pub fn with_config(config: FoundryLocalChatConfig) -> Self {
        Self {
            config,
            model: OnceCell::new(),
            history: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    async fn ensure_model(&self) -> Result<Arc<Model>, Error> {
        let alias = self.config.resolve_model_alias()?.to_string();
        let snap = self.config.provider_snapshot();
        let model = self
            .model
            .get_or_try_init(|| async {
                provider().ensure_model(&snap, &alias).await
            })
            .await?;
        Ok(model.clone())
    }

    fn configure_client(&self, mut client: ChatClient) -> ChatClient {
        if let Some(v) = self.config.temperature {
            client = client.temperature(v as f64);
        }
        if let Some(v) = self.config.max_tokens {
            client = client.max_tokens(v);
        }
        if let Some(v) = self.config.top_p {
            client = client.top_p(v as f64);
        }
        if let Some(v) = self.config.top_k {
            client = client.top_k(v);
        }
        if let Some(v) = self.config.frequency_penalty {
            client = client.frequency_penalty(v as f64);
        }
        if let Some(v) = self.config.presence_penalty {
            client = client.presence_penalty(v as f64);
        }
        if let Some(v) = self.config.n {
            client = client.n(v);
        }
        if let Some(v) = self.config.random_seed {
            client = client.random_seed(v);
        }
        if let Some(ref fmt) = self.config.response_format {
            client = client.response_format(fmt.clone().into_sdk());
        }
        if let Some(ref tc) = self.config.tool_choice {
            client = client.tool_choice(tc.clone().into_sdk());
        }
        client
    }

    fn append_history(&self, session_id: &str, entry: HistoryEntry) {
        self.history
            .lock()
            .entry(session_id.to_string())
            .or_insert_with(Vec::new)
            .push(entry);
    }

    fn extend_history(&self, session_id: &str, entries: Vec<HistoryEntry>) {
        if entries.is_empty() {
            return;
        }
        self.history
            .lock()
            .entry(session_id.to_string())
            .or_insert_with(Vec::new)
            .extend(entries);
    }

    fn build_message_values(&self, session_id: &str, user_message: &Value) -> Vec<Value> {
        let mut messages: Vec<Value> = Vec::new();
        if let Some(ref sys) = self.config.system_prompt {
            messages.push(serde_json::json!({"role": "system", "content": sys}));
        }
        {
            let hist = self.history.lock();
            if let Some(entries) = hist.get(session_id) {
                let start = window_start(entries, self.config.history_turns);
                for entry in entries.iter().skip(start) {
                    messages.push(entry.message.clone());
                }
            }
        }
        messages.push(user_message.clone());
        messages
    }

    fn decode_messages(values: Vec<Value>) -> Result<Vec<ChatCompletionRequestMessage>, Error> {
        serde_json::from_value(Value::Array(values)).map_err(|e| {
            Error::Execution(format!(
                "[foundry-local] failed to decode messages for Foundry SDK: {}",
                e
            ))
        })
    }

    fn decode_tools(tools: &[ToolSpec]) -> Result<Vec<ChatCompletionTools>, Error> {
        if tools.is_empty() {
            return Ok(Vec::new());
        }
        serde_json::from_value(to_openai_tools_array(tools)).map_err(|e| {
            Error::Execution(format!(
                "[foundry-local] failed to encode tools for Foundry SDK: {}",
                e
            ))
        })
    }

    fn extract_user_text(data: &RuntimeData) -> Result<Option<String>, Error> {
        match data {
            RuntimeData::Text(t) => {
                if t.trim().is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(t.clone()))
                }
            }
            RuntimeData::Json(j) => match j.get("content").or(j.get("text")).and_then(|v| v.as_str()) {
                Some(s) if !s.is_empty() => Ok(Some(s.to_string())),
                _ => {
                    tracing::debug!(
                        node = "FoundryLocalChatNode",
                        "Dropping JSON input with no `content`/`text` field"
                    );
                    Ok(None)
                }
            },
            other => Err(Error::Execution(format!(
                "FoundryLocalChatNode expects Text or Json input, got: {}",
                other.data_type()
            ))),
        }
    }

    async fn run_chat<F>(
        &self,
        session_id: &str,
        user_message: Value,
        callback: &mut F,
    ) -> Result<usize, Error>
    where
        F: FnMut(RuntimeData) -> Result<(), Error> + Send,
    {
        let model = self.ensure_model().await?;
        let tool_registry = self.config.build_tool_registry();
        let tools_sdk = Self::decode_tools(&tool_registry)?;
        let tools_arg: Option<&[ChatCompletionTools]> = if tools_sdk.is_empty() {
            None
        } else {
            Some(tools_sdk.as_slice())
        };

        let message_values = self.build_message_values(session_id, &user_message);
        self.append_history(session_id, HistoryEntry::user_message(user_message));
        let messages = Self::decode_messages(message_values)?;

        let output_channel = self
            .config
            .output_channel
            .clone()
            .unwrap_or_else(|| TEXT_CHANNEL_DEFAULT.to_string());

        let client = self.configure_client(model.create_chat_client());

        if self.config.streaming {
            self.run_streaming(
                session_id,
                client,
                &messages,
                tools_arg,
                &tool_registry,
                &output_channel,
                callback,
            )
            .await
        } else {
            self.run_blocking(
                session_id,
                client,
                &messages,
                tools_arg,
                &output_channel,
                callback,
            )
            .await
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_streaming<F>(
        &self,
        session_id: &str,
        client: ChatClient,
        messages: &[ChatCompletionRequestMessage],
        tools: Option<&[ChatCompletionTools]>,
        tool_registry: &[ToolSpec],
        output_channel: &str,
        callback: &mut F,
    ) -> Result<usize, Error>
    where
        F: FnMut(RuntimeData) -> Result<(), Error> + Send,
    {
        let mut stream = client
            .complete_streaming_chat(messages, tools)
            .await
            .map_err(|e| {
                Error::Execution(format!(
                    "[foundry-local] streaming chat request failed: {}",
                    e
                ))
            })?;

        let mut full_text = String::new();
        let mut token_count = 0usize;
        let mut tool_calls: HashMap<u64, ToolCallAccum> = HashMap::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| {
                Error::Execution(format!(
                    "[foundry-local] stream read error from Foundry Core: {}",
                    e
                ))
            })?;

            let Some(choice) = chunk.choices.first() else {
                continue;
            };

            if let Some(content) = &choice.delta.content {
                if !content.is_empty() {
                    full_text.push_str(content);
                    token_count += 1;
                    callback(RuntimeData::Text(tag_text_str(content, output_channel)))?;
                }
            }

            if let Some(tool_call_deltas) = &choice.delta.tool_calls {
                for tcd in tool_call_deltas {
                    let entry = tool_calls.entry(tcd.index as u64).or_default();
                    if let Some(id) = &tcd.id {
                        entry.id = id.clone();
                    }
                    if let Some(func) = &tcd.function {
                        if let Some(name) = &func.name {
                            entry.name = name.clone();
                        }
                        if let Some(args) = &func.arguments {
                            entry.arguments.push_str(args);
                        }
                    }
                }
            }
        }

        let mut indices: Vec<u64> = tool_calls.keys().copied().collect();
        indices.sort_unstable();
        tracing::info!(
            node = "FoundryLocalChatNode",
            tokens = token_count,
            chars = full_text.len(),
            tool_calls_received = indices.len(),
            "[foundry-local] streaming complete"
        );

        let mut tool_calls_for_history: Vec<Value> = Vec::with_capacity(indices.len());
        let mut tool_result_entries: Vec<HistoryEntry> = Vec::with_capacity(indices.len());

        for idx in indices {
            let entry = match tool_calls.remove(&idx) {
                Some(e) => e,
                None => continue,
            };
            if entry.name.is_empty() {
                continue;
            }
            let call_id = if entry.id.is_empty() {
                format!("call_{}", idx)
            } else {
                entry.id.clone()
            };
            tool_calls_for_history.push(serde_json::json!({
                "id": call_id,
                "type": "function",
                "function": { "name": entry.name, "arguments": entry.arguments },
            }));
            tool_result_entries.push(HistoryEntry::tool_result(&call_id, &entry.name, ""));
            dispatch_tool_call(tool_registry, &entry, output_channel, callback)?;
        }

        let mut turn_entries: Vec<HistoryEntry> = Vec::new();
        if !tool_calls_for_history.is_empty() {
            turn_entries.push(HistoryEntry::assistant_with_tool_calls(
                if full_text.is_empty() {
                    None
                } else {
                    Some(&full_text)
                },
                Value::Array(tool_calls_for_history),
            ));
            turn_entries.extend(tool_result_entries);
        } else if !full_text.is_empty() {
            turn_entries.push(HistoryEntry::assistant_text(&full_text));
        }
        self.extend_history(session_id, turn_entries);

        // Synthesize the end-of-turn sentinel so the
        // `ConversationCoordinatorNode` downstream flips to Idle and
        // emits its `turn_state` envelope. Matches what the in-tree
        // `OpenAIChatNode` does for OpenAI-shaped providers that
        // don't emit `<|text_end|>` themselves.
        callback(RuntimeData::Text(tag_text_str("<|text_end|>", output_channel)))?;
        Ok(token_count)
    }

    async fn run_blocking<F>(
        &self,
        session_id: &str,
        client: ChatClient,
        messages: &[ChatCompletionRequestMessage],
        tools: Option<&[ChatCompletionTools]>,
        output_channel: &str,
        callback: &mut F,
    ) -> Result<usize, Error>
    where
        F: FnMut(RuntimeData) -> Result<(), Error> + Send,
    {
        let response = client.complete_chat(messages, tools).await.map_err(|e| {
            Error::Execution(format!(
                "[foundry-local] blocking chat request failed: {}",
                e
            ))
        })?;

        let Some(choice) = response.choices.into_iter().next() else {
            return Ok(0);
        };
        let content = choice.message.content.unwrap_or_default();
        if content.is_empty() {
            return Ok(0);
        }
        self.append_history(session_id, HistoryEntry::assistant_text(&content));
        callback(RuntimeData::Text(tag_text_str(&content, output_channel)))?;
        callback(RuntimeData::Text(tag_text_str("<|text_end|>", output_channel)))?;
        Ok(1)
    }
}

#[async_trait]
impl AsyncStreamingNode for FoundryLocalChatNode {
    fn node_type(&self) -> &str {
        "FoundryLocalChatNode"
    }

    async fn initialize(&self, _ctx: &dyn InitializeContextRead) -> Result<(), Error> {
        let alias = self.config.resolve_model_alias()?.to_string();
        tracing::info!(
            node = "FoundryLocalChatNode",
            app_name = %self.config.app_name,
            model_alias = %alias,
            streaming = self.config.streaming,
            history_turns = self.config.history_turns,
            prewarm = self.config.prewarm,
            "Initializing FoundryLocalChatNode"
        );
        if self.config.prewarm {
            // Force-load the model now so the first request is fast.
            let _ = self.ensure_model().await?;
        }
        Ok(())
    }

    async fn process(&self, data: RuntimeData) -> Result<RuntimeData, Error> {
        // Single-shot path: collect streaming output, return last frame.
        let mut outputs: Vec<RuntimeData> = Vec::new();
        let session_id = "default".to_string();
        let user_text = match Self::extract_user_text(&data)? {
            Some(t) => t,
            None => {
                return Ok(RuntimeData::Text(tag_text_str("", TEXT_CHANNEL_DEFAULT)));
            }
        };
        let user_msg = serde_json::json!({"role": "user", "content": user_text});
        let mut cb = |out: RuntimeData| -> Result<(), Error> {
            outputs.push(out);
            Ok(())
        };
        self.run_chat(&session_id, user_msg, &mut cb).await?;
        outputs
            .into_iter()
            .last()
            .ok_or_else(|| Error::Execution("FoundryLocalChatNode: no output generated".into()))
    }

    async fn process_streaming<F>(
        &self,
        data: RuntimeData,
        session_id: Option<String>,
        mut callback: F,
    ) -> Result<usize, Error>
    where
        F: FnMut(RuntimeData) -> Result<(), Error> + Send,
    {
        let sid = session_id.unwrap_or_else(|| "default".to_string());
        let user_text = match Self::extract_user_text(&data)? {
            Some(t) => t,
            None => return Ok(0),
        };
        let user_msg = serde_json::json!({"role": "user", "content": user_text});
        self.run_chat(&sid, user_msg, &mut callback).await
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct FoundryLocalChatNodeFactory;

impl FfiNodeFactory for FoundryLocalChatNodeFactory {
    fn node_type(&self) -> RString {
        RString::from("FoundryLocalChatNode")
    }

    fn create(&self, params: RString) -> RResult<FfiNodeBox, RString> {
        let params: Value =
            serde_json::from_str(params.as_str()).unwrap_or(serde_json::json!({}));
        let config: FoundryLocalChatConfig =
            serde_json::from_value(params).unwrap_or_default();
        // Surface the missing-alias error at create time rather than
        // at first request — the manifest author gets immediate
        // feedback when the pipeline graph is built.
        if let Err(e) = config.resolve_model_alias() {
            return RErr(RString::from(format!("{}", e)));
        }
        let node = FoundryLocalChatNode::with_config(config);
        ROk(FfiNode_TO::from_value(
            StreamingNodeFfiAdapter::new(node),
            TD_Opaque,
        ))
    }
}
