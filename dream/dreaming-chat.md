# dreaming-chat.md — Chat Deep-Dive for the Dreaming Feature

## 1. Executor Turn Flow End-to-End

**Entry point:** `routes.py:793` — `POST /sessions/{session_id}/stream` (`stream_chat_post`)

The HTTP handler does the following in order:
1. Validates the session belongs to the user (`_validate_and_get_session`, `routes.py:833`).
2. If a turn is already in-flight (`is_turn_in_flight`), drops the message into a Redis pending buffer and returns HTTP 202 — no enqueue.
3. Persists the user message via `append_and_save_message` (`model.py:650`), which acquires a Redis NX lock, checks for duplicates, writes to Postgres, and back-fills `sequence` on the in-memory object.
4. Calls `enqueue_copilot_turn` (`executor/utils.py:192`) which serializes a `CoPilotExecutionEntry` and publishes it to RabbitMQ exchange `copilot_execution` (DIRECT, routing key `copilot.run`).
5. Calls `stream_registry.create_session` to write the `chat:task:meta:{session_id}` Redis hash (status=running, turn_id, etc.).
6. Subscribes via `stream_registry.subscribe_to_session` and streams chunks from the Redis Stream `chat:stream:{turn_id}` back to the browser as SSE.
7. SSE heartbeats fire every ~5s from `_stream_listener` (`stream_registry.py:559`).

**Consumer side:** `executor/manager.py` — `CoPilotExecutor` (a separate container/process).
- Two threads: `_consume_run` (DIRECT queue, `prefetch_count=pool_size`) and `_consume_cancel` (FANOUT queue, auto-ack).
- On a run message: acquires a Redis cluster lock (`copilot:session:{session_id}:lock`), submits `execute_copilot_turn` to a `ThreadPoolExecutor` (`pool_size = settings.config.num_copilot_workers`).
- **Tool calls execute in-process** inside the worker thread's asyncio loop — there is no separate sandbox for tool dispatch itself. The E2B sandbox (`e2b_sandbox.py`) is a remote cloud VM used for `bash_exec` commands only; all other tools run directly in the executor process.
- `execute_copilot_turn` (`processor.py:90`) → `CoPilotProcessor.execute` → `_execute_async` → picks either `stream_chat_completion_sdk` (Claude Agent SDK via subprocess CLI) or `stream_chat_completion_baseline` (OpenAI-compatible HTTP), then drives `stream_registry.stream_and_publish` which writes every chunk to `XADD chat:stream:{turn_id}`.
- **Timeout:** RabbitMQ queue is configured with `x-consumer-timeout = 3600000 ms` (1 hour). Graceful shutdown timeout is 30 minutes. The SDK path has its own `claude_agent_max_budget_usd = $10` and `claude_agent_max_turns = 50` per turn. There is no HTTP-level timeout on the SSE stream itself; the frontend reconnects on disconnect.
- **Retry:** transient API errors (429, 5xx) are retried up to `claude_agent_max_transient_retries = 3` times inside `sdk/service.py`. Context-too-large errors get up to `_MAX_STREAM_ATTEMPTS = 3` retries with progressively smaller transcript. RabbitMQ nack+requeue is used only for infrastructure failures (can't acquire lock, pool full); completed turns (success or error) are acked and not requeued.

Key files: `autogpt_platform/backend/backend/api/features/chat/routes.py`, `autogpt_platform/backend/backend/copilot/executor/manager.py`, `autogpt_platform/backend/backend/copilot/executor/processor.py`, `autogpt_platform/backend/backend/copilot/executor/utils.py`, `autogpt_platform/backend/backend/copilot/stream_registry.py`.

## 2. SSE Event Taxonomy

All event types are defined in `autogpt_platform/backend/backend/copilot/response_model.py`.

The `ResponseType` enum (`response_model.py:22`) defines every type:

| Type string | Python class | Wire format | Purpose |
|---|---|---|---|
| `start` | `StreamStart` | `data: {...}` | Beginning of a new assistant message; carries `messageId` |
| `start-step` | `StreamStartStep` | `data: {...}` | One LLM API call begins (for multi-step tool loops) |
| `finish-step` | `StreamFinishStep` | `data: {...}` | One LLM API call ends |
| `text-start` | `StreamTextStart` | `data: {...}` | Text block opens; carries `id` |
| `text-delta` | `StreamTextDelta` | `data: {...}` | Streaming text fragment; carries `id` + `delta` |
| `text-end` | `StreamTextEnd` | `data: {...}` | Text block closes |
| `reasoning-start` | `StreamReasoningStart` | `data: {...}` | Extended thinking block opens |
| `reasoning-delta` | `StreamReasoningDelta` | `data: {...}` | Thinking fragment |
| `reasoning-end` | `StreamReasoningEnd` | `data: {...}` | Thinking block closes |
| `tool-input-start` | `StreamToolInputStart` | `data: {...}` | Tool call begins; carries `toolCallId` + `toolName` |
| `tool-input-available` | `StreamToolInputAvailable` | `data: {...}` | Full tool input JSON ready |
| `tool-output-available` | `StreamToolOutputAvailable` | `data: {...}` | Tool result; carries `toolCallId` + `output` |
| `error` | `StreamError` | `data: {...}` | Error; carries `errorText` |
| `usage` | `StreamUsage` | `: usage {...}` | SSE comment (ignored by AI SDK parser) |
| `heartbeat` | `StreamHeartbeat` | `: heartbeat` | SSE comment; keeps proxy alive |
| `status` | `StreamStatus` | `data: {...}` | Transient human-readable status notification |
| `finish` | `StreamFinish` | `data: {...}` | Stream end marker |

**Adding a new event type:** Three sites must be updated:
1. Add the string to `ResponseType` enum — `response_model.py:22`.
2. Create a new `StreamXxx(StreamBaseResponse)` dataclass in `response_model.py`.
3. Register the class in `_reconstruct_chunk`'s `type_to_class` dict — `stream_registry.py:1060`. This is the only place where unknown types produce a `logger.warning` and return `None` (silently dropped), so forgetting this causes dream events to be swallowed on SSE replay.

**Dream event plug-in:** Add `DREAM_STARTED = "dream-started"`, `DREAM_STEP = "dream-step"`, `DREAM_FINISHED = "dream-finished"`, `DREAM_MEMORY_PROPOSED = "dream-memory-proposed"` to `ResponseType`, their payload classes, and register them in `_reconstruct_chunk`. The `StreamStatus` class is the closest existing analogue; dream events would be similar but carry structured `memory_id`/`step_index`/`session_id` fields.

## 3. Tool-Call Surface Available to the Chat Agent

All tools are registered in `autogpt_platform/backend/backend/copilot/tools/__init__.py:62` (`TOOL_REGISTRY`). Every tool inherits from `BaseTool` (`tools/base.py`) and implements `name`, `description`, `parameters`, `requires_auth`, and `_execute(user_id, session, **kwargs)`.

| Tool name | File | What it does | Auth required |
|---|---|---|---|
| `memory_store` | `graphiti_store.py` | Enqueues a `MemoryEnvelope` episode to the user's Graphiti graph | Yes |
| `memory_search` | `graphiti_search.py` | Semantic search over the user's fact graph + last 5 episodes | Yes |
| `memory_forget_search` | `graphiti_forget.py` | Returns candidate edges for deletion by UUID | Yes |
| `memory_forget_confirm` | `graphiti_forget.py` | Soft or hard deletes edges by UUID | Yes |
| `add_understanding` | `add_understanding.py` | Updates user's `BusinessUnderstanding` Postgres record | Yes |
| `ask_question` | `ask_question.py` | Emits a clarifying-question block to the SSE stream | No |
| `create_agent` | `create_agent.py` | Creates a new agent graph | Yes |
| `customize_agent` | `customize_agent.py` | Modifies agent graph properties | Yes |
| `edit_agent` | `edit_agent.py` | Edits agent graph nodes/edges | Yes |
| `find_agent` | `find_agent.py` | Searches user's library for agents | Yes |
| `find_block` | `find_block.py` | Searches block registry | No |
| `find_library_agent` | `find_library_agent.py` | Searches library | Yes |
| `run_agent` | `run_agent.py` | Executes an agent graph | Yes |
| `run_block` | `run_block.py` | Executes a single block | Yes |
| `continue_run_block` | `continue_run_block.py` | Continues a paused block execution | Yes |
| `run_sub_session` | `run_sub_session.py` | Spawns a sub-AutoPilot session (recursive) | Yes |
| `get_sub_session_result` | `get_sub_session_result.py` | Polls sub-session result | Yes |
| `run_mcp_tool` | `run_mcp_tool.py` | Calls an external MCP server tool | Yes |
| `get_mcp_guide` | `get_mcp_guide.py` | Returns MCP configuration guide | No |
| `view_agent_output` | `agent_output.py` | Retrieves agent execution output | Yes |
| `search_docs` | `search_docs.py` | Semantic search over platform docs | No |
| `get_doc_page` | `get_doc_page.py` | Fetches a doc page by path | No |
| `get_agent_building_guide` | `get_agent_building_guide.py` | Returns agent building guide | No |
| `web_fetch` | `web_fetch.py` | Fetches a URL safely | No |
| `browser_navigate` | `agent_browser.py` | Browser automation: navigate | Conditional |
| `browser_act` | `agent_browser.py` | Browser automation: click/type | Conditional |
| `browser_screenshot` | `agent_browser.py` | Browser automation: screenshot | Conditional |
| `bash_exec` | `bash_exec.py` | Runs bash in E2B sandbox or bubblewrap | No |
| `connect_integration` | `connect_integration.py` | Initiates OAuth connection flow | Yes |
| folder/feature-request/workspace-file/validate-fix tools | various | CRUD + validation utilities | Yes |

**Tool availability is evaluated per-request** via `tool.is_available` (`get_available_tools()`, `__init__.py:122`), which checks env-var and binary prerequisites. Browser tools are excluded when the agent-browser CLI is not installed.

**For the dream pass:** The dream pass needs a restricted tool surface. It should only have access to `memory_search`, `memory_store`, and `memory_forget_confirm` (and possibly a new `memory_status_update` for flipping `status` to `superseded`). A `permissions: CopilotPermissions` field already exists on `CoPilotExecutionEntry` (`executor/utils.py:167`) and is enforced by `apply_tool_permissions` (`tools/permissions.py`) in the SDK path. Passing a dream-specific `CopilotPermissions` object when enqueuing will restrict the tool surface without code changes in the tool implementations.

## 4. System Prompt Assembly

**File:** `autogpt_platform/backend/backend/copilot/service.py`

The system prompt is static and identical for all users, enabling cross-session LLM prompt caching. User-specific context is injected into the first user message instead.

**`_build_system_prompt(user_id)` — `service.py:336`:**
1. Fetches `BusinessUnderstanding` from Postgres for the user (via `understanding_db().get_business_understanding(user_id)`).
2. Fetches the prompt text from Langfuse (`_fetch_langfuse_prompt()`, `service.py:296`) with a 5-minute in-process cache. Falls back to `_CACHEABLE_SYSTEM_PROMPT` (`service.py:100`) if Langfuse is unconfigured or fails.
3. Returns `(prompt_text, understanding)`.

The static `_CACHEABLE_SYSTEM_PROMPT` constant instructs the LLM to parse three server-injected blocks: `<user_context>`, `<memory_context>`, and `<env_context>`.

**`inject_user_context(understanding, message, session_id, session_messages, warm_ctx, env_ctx)` — `service.py:360`:**
Called on the first turn only. Assembles the first user message as:
```
<memory_context>\n{warm_ctx}\n</memory_context>\n\n
<env_context>\n{env_ctx}\n</env_context>\n\n
<user_context>\n{sanitized_understanding}\n</user_context>\n\n
{sanitized_user_message}
```
All user-controlled fields are sanitized via `sanitize_user_supplied_context` (strips attacker-supplied tags) and `_sanitize_user_context_field` (HTML-escapes `<`/`>`). The assembled content is persisted to the DB so page-reload replays see the same prefixed message.

**`<memory_context>` content — `graphiti/context.py:20` (`fetch_warm_context`):**
- Triggered on first turn, with a 5-second timeout (`graphiti_config.context_timeout`).
- Calls `client.search(query=message, num_results=graphiti_config.context_max_facts)` + `client.retrieve_episodes(last_n=5)`.
- Formats as `<temporal_context><FACTS>...</FACTS><RECENT_EPISODES>...</RECENT_EPISODES></temporal_context>`.
- No fixed token budget — the number of facts is capped by `context_max_facts` (config field on `GraphitiConfig`). Non-global-scope episodes are filtered out.
- On timeout or error: returns `None`, warm context is omitted gracefully.

**"This is a dream pass" system prompt:** Would go in `_build_system_prompt` or its Langfuse equivalent. The cleanest approach is a new Langfuse prompt name (e.g. `CoPilot Dream Prompt`) selected when the `CoPilotExecutionEntry` carries a `dream_pass=True` flag, or simply a separate entry point function `_build_dream_system_prompt` that returns a static string with consolidation/recombination instructions, bypassing Langfuse.

## 5. Authoring Messages Programmatically

**`ChatMessage` model — `model.py:58`:**
```python
class ChatMessage(BaseModel):
    role: str           # "user", "assistant", "system", "tool"
    content: str | None
    tool_calls: list[dict] | None
    sequence: int | None
    duration_ms: int | None
    # ... name, tool_call_id, refusal, function_call
```

The `role` field is a plain `str` with no enum constraint in the Pydantic model. The Prisma schema mirrors this: `ChatMessage.role` is a `String`. So any string value is accepted — `"dream"` could be used, or `"assistant"` with `metadata` tagging. However, the downstream SSE replay path (`routes.py`, `_strip_injected_context`) and the AI SDK's `useChat` expect `role` values in `{user, assistant, system, tool}`. An unknown role would be passed through to the frontend as-is; the AI SDK would likely render it as plain text without special treatment.

**Inserting a dream summary:** Call `append_and_save_message(session_id, ChatMessage(role="assistant", content="[DREAM SUMMARY] ..."))` from the dream pass job. This uses the Redis lock, idempotency check, and DB write path exactly as live turns do. The sequence number is assigned automatically.

**Role recommendation:** Use `role="assistant"` with a content prefix like `[dream]` or store the dream in a separate `ChatSession` with `metadata.kind = "dream"` (extend `ChatSessionMetadata`). A new role value like `"dream"` would not break Postgres or Python but would render incorrectly in the AI SDK frontend. Using `role="assistant"` with a detectable prefix is safer and requires no frontend changes for basic display.

**Prisma schema constraint:** `schema.prisma:226–275` — `ChatMessage.role` is `String` (no enum), so no migration is needed to write arbitrary role values.

**SSE replay path:** The `GET /sessions/{session_id}` endpoint returns all messages with `role` passed through unchanged. The SSE replay (`subscribe_to_session`) replays turn-scoped Redis Stream events, not DB messages; dream messages written directly to the DB would appear in the `GET` response but not in an SSE replay unless explicitly published to a stream.

## 6. Session Lifecycle Hooks

There are no explicit lifecycle hooks (callbacks or event bus entries) fired on session events. Lifecycle-adjacent behaviors:

- **Session create:** `create_chat_session` (`model.py:715`) — writes to Postgres + Redis cache. No hooks; just a synchronous create.
- **First message:** The `inject_user_context` call on turn 1 is the only first-message-specific code. Graphiti warm context is also fetched only on the first turn (detected by checking `session_messages` for existing user messages).
- **Last message / turn end:** `mark_session_completed` (`stream_registry.py:798`) — sets `status` to `completed` or `failed` in Redis, publishes `StreamFinish`, fires a WebSocket `copilot_completion` notification via `_notification_bus`. Session title generation (`_update_title_async`) is fired from within the SDK/baseline service after the first user message is processed.
- **Inactivity:** Nothing. There is no TTL-based wake or inactivity trigger anywhere. The session `chat:task:meta:{session_id}` Redis key has a 1-hour TTL (`stream_ttl`), but expiry is not observed by any job — it just makes the key disappear.
- **Session delete:** `delete_chat_session` (`model.py:770`) — deletes from Postgres, removes Redis cache key, shuts down local browser daemon (best-effort). No hook.

**For dream triggering:** There is no hook to attach to. The dream scheduler job would need to query Postgres for sessions updated before a cutoff time using `get_user_sessions` + filtering by `updated_at`, rather than subscribing to a lifecycle event.

## 7. Mode/Model Selection

**Two orthogonal axes:**

**Axis 1 — execution path (`mode`):**
`CopilotMode = Literal["fast", "extended_thinking"]`
- `fast` → `stream_chat_completion_baseline` (direct OpenAI-compatible HTTP to OpenRouter)
- `extended_thinking` → `stream_chat_completion_sdk` (Claude Agent SDK subprocess CLI)
- `None` → resolved by LaunchDarkly flag `COPILOT_SDK` + `use_claude_code_subscription` config, then `config.use_claude_agent_sdk` (default `True`).

Gate: `resolve_effective_mode` (`processor.py:36`) checks the `CHAT_MODE_OPTION` LaunchDarkly flag server-side. If the user isn't entitled, `mode` is stripped to `None`.

**Axis 2 — model tier (`model`):**
`CopilotLlmModel = Literal["standard", "advanced"]`
- `standard` → `config.model` (default `anthropic/claude-sonnet-4-6`)
- `advanced` → `config.advanced_model` (default `anthropic/claude-opus-4-7`)
- `None` → `config.model`

Resolution function: `resolve_chat_model(tier)` in `service.py:44`. Used by both baseline and SDK paths.

**SDK model resolution** is slightly different — `_resolve_sdk_model()` (`sdk/service.py:700`) strips the OpenRouter provider prefix (e.g. `anthropic/claude-sonnet-4-6` → `claude-sonnet-4-6`) to get the bare model name for the CLI. The `_resolve_model_and_multiplier` function (`sdk/service.py:728`) applies a 5× cost multiplier for Opus (rate-limit accounting).

**Adding a "dream" preset:**
- Add `"dream"` to `CopilotMode` (or use a separate field `dream_mode: bool` on `CoPilotExecutionEntry`).
- In `resolve_use_sdk_for_mode` (`processor.py:59`): add `if mode == "dream": return True` (always use SDK path for dreams).
- In `_resolve_model_and_multiplier` or `resolve_chat_model`: add `if model == "dream": return config.dream_model` where `dream_model` is a new `ChatConfig` field (e.g. `anthropic/claude-sonnet-4-6` with higher temperature).
- Temperature is currently not exposed per-turn — it's a fixed parameter inside the baseline service's `client.chat.completions.create` call. For the SDK path, the CLI does not expose a `--temperature` flag directly; temperature would need to be applied via a system prompt instruction ("respond with high creativity and broad associations") rather than a model parameter.

## 8. Frontend Coupling

**`useCopilotStream.ts`** (`copilot/useCopilotStream.ts`):
- Uses `@ai-sdk/react` `useChat` with a `DefaultChatTransport` pointing at `POST /api/chat/sessions/{sessionId}/stream` (line 71).
- `resumeStream` calls `GET /api/chat/sessions/{sessionId}/stream`.
- Handles `onFinish`, `onError`, wake re-sync, and reconnect logic.
- **Integration seam:** Dream events arriving as `data:` SSE events with unknown `type` values are passed to the AI SDK's stream parser. The AI SDK uses `z.strictObject()` for known types and logs a console warning + skips unknown types — so dream events would be silently dropped unless the frontend explicitly handles them.
- To surface dream events, the `DefaultChatTransport` approach is insufficient — dream events need a side-channel or a custom transport that intercepts unknown event types before the AI SDK parser sees them. The cleanest approach is to intercept at the `EventSource`/fetch level before the `useChat` hook, or to use the `StreamStatus` event type (already rendered by the frontend as a transient notification) as a carrier for lightweight dream progress until a dedicated UI exists.

**`useCopilotPage.ts`** (`copilot/useCopilotPage.ts`):
- Manages session selection, message hydration, and the `useCopilotStream` integration.
- `hydratedMessages` comes from `GET /sessions/{sessionId}` (React Query). Dream messages written to the DB as `role="assistant"` would appear in `hydratedMessages` automatically after the next refetch.
- `hasActiveStream` comes from `session.active_stream !== null` in the API response — a dream pass running as its own session would need a separate `sessionId` to avoid conflating with the user's active session.

**Key integration seams for dream events:**
1. `_reconstruct_chunk` in `stream_registry.py:1050` — must know dream event classes to replay them.
2. The AI SDK stream parser in `useCopilotStream.ts` — must not swallow dream events; requires a custom transport or event interceptor.
3. The session `GET` endpoint (`routes.py`) — dream messages appear automatically if written to the DB under the same session.
4. `useCopilotNotifications.ts` — already listens to WebSocket `copilot_completion` events; a `dream_completed` notification type could be added to the same bus (`_notification_bus`, `stream_registry.py:899`).

---

## Where Dream Events Plug In — Edit Checklist

Specific files and locations that must be touched to wire up a dream pass with its own stream events:

**Backend — event schema:**
- `autogpt_platform/backend/backend/copilot/response_model.py:22` — add `DREAM_STARTED`, `DREAM_STEP`, `DREAM_FINISHED`, `DREAM_MEMORY_PROPOSED` to `ResponseType` enum.
- `response_model.py` (after line 316) — add `StreamDreamStarted`, `StreamDreamStep`, `StreamDreamFinished`, `StreamDreamMemoryProposed` dataclasses.

**Backend — stream replay:**
- `autogpt_platform/backend/backend/copilot/stream_registry.py:1060` — add dream event classes to `type_to_class` dict inside `_reconstruct_chunk`.

**Backend — mode/model routing:**
- `autogpt_platform/backend/backend/copilot/config.py` — add `dream_model: str` field to `ChatConfig`.
- `autogpt_platform/backend/backend/copilot/executor/processor.py:59` — extend `resolve_use_sdk_for_mode` to handle `mode == "dream"`.
- `autogpt_platform/backend/backend/copilot/service.py:44` — extend `resolve_chat_model` with `"dream"` tier.

**Backend — system prompt:**
- `autogpt_platform/backend/backend/copilot/service.py:336` — add `_build_dream_system_prompt()` or extend `_build_system_prompt` to branch on a dream flag.

**Backend — session metadata:**
- `autogpt_platform/backend/backend/copilot/model.py:48` — extend `ChatSessionMetadata` with `last_dream_at: datetime | None`, `dream_enabled: bool`, `dream_config: dict | None`.

**Backend — scheduler:**
- `autogpt_platform/backend/backend/executor/scheduler.py` (after existing `@expose` methods, ~line 660) — add `add_dream_pass_schedule(user_id, user_timezone)` and `_execute_dream_pass(user_id)`.

**Backend — tool permissions for dream:**
- `autogpt_platform/backend/backend/copilot/executor/utils.py:222` — pass a dream-specific `CopilotPermissions` object when calling `enqueue_copilot_turn` from the dream job.

**Frontend — event handling:**
- `autogpt_platform/frontend/src/app/(platform)/copilot/useCopilotStream.ts:67` — intercept unknown SSE event types before they reach the AI SDK parser, or handle `StreamStatus`-carried dream progress.
- `autogpt_platform/frontend/src/app/(platform)/copilot/useCopilotNotifications.ts` — add handler for `dream_completed` WebSocket event type.
- `autogpt_platform/frontend/src/app/(platform)/copilot/store.ts` — add dream state (if surfacing dream progress in the UI is needed).

---

## Key File References

- `autogpt_platform/backend/backend/api/features/chat/routes.py` — HTTP entry points, SSE streaming, enqueue call
- `autogpt_platform/backend/backend/copilot/executor/utils.py` — `CoPilotExecutionEntry`, `enqueue_copilot_turn`, queue constants and timeouts
- `autogpt_platform/backend/backend/copilot/executor/manager.py` — RabbitMQ consumer, thread pool, cluster lock
- `autogpt_platform/backend/backend/copilot/executor/processor.py` — `execute_copilot_turn`, mode routing, stream loop
- `autogpt_platform/backend/backend/copilot/stream_registry.py` — Redis Stream publish/subscribe, `_reconstruct_chunk`, `mark_session_completed`
- `autogpt_platform/backend/backend/copilot/response_model.py` — all SSE event types and their wire shapes
- `autogpt_platform/backend/backend/copilot/service.py` — system prompt, `inject_user_context`, `resolve_chat_model`
- `autogpt_platform/backend/backend/copilot/config.py` — `ChatConfig`, `CopilotMode`, `CopilotLlmModel`
- `autogpt_platform/backend/backend/copilot/model.py` — `ChatSession`, `ChatMessage`, `ChatSessionMetadata`, `append_and_save_message`
- `autogpt_platform/backend/backend/copilot/tools/__init__.py` — `TOOL_REGISTRY`, `execute_tool`, `get_available_tools`
- `autogpt_platform/backend/backend/copilot/graphiti/context.py` — `fetch_warm_context`, warm context formatting
- `autogpt_platform/backend/backend/copilot/graphiti/ingest.py` — `enqueue_episode`, `_ingestion_worker`
- `autogpt_platform/backend/backend/copilot/graphiti/memory_model.py` — `MemoryEnvelope`, `MemoryStatus`, `SourceKind`
- `autogpt_platform/backend/backend/copilot/sdk/service.py` — SDK path, `_resolve_model_and_multiplier`, `_resolve_sdk_model`
- `autogpt_platform/frontend/src/app/(platform)/copilot/useCopilotStream.ts` — AI SDK transport, reconnect logic, event handling
- `autogpt_platform/frontend/src/app/(platform)/copilot/useCopilotPage.ts` — session management, hydration orchestration
