# Local LLM Routing

> **Status**: Draft v0.1 — wire spec stable, platform-side routing
> behind LD flag `LOCAL_LLM_ROUTING` (default off).

Local LLM routing lets the platform send a completion turn to a model
running on the user's own machine (via the shim) instead of Anthropic
or OpenRouter. The user keeps the prompt and response inside their own
process — the cloud platform never sees the content, only a metadata
record in the audit log.

This doc covers the wire spec (briefly — full spec in
[PROTOCOL.md](PROTOCOL.md#local-llm)), the activation gating, the
backend matrix, per-OS install notes, and privacy considerations.

---

## What you buy

| Benefit | Why local LLM helps |
|---|---|
| **Privacy** | Prompts and responses never transit the AutoGPT platform or the cloud LLM provider. Pairs with [PRIVACY_MODE.md](PRIVACY_MODE.md) (file content also stays local). |
| **Cost** | No per-token charge. Useful for the "fast" tier where the user is bouncing many short turns off a small model. |
| **Latency for small models** | Llama 3.2:3b on a recent M-series Mac runs faster than a cloud round-trip for short responses. Loses on long completions where the GPU horsepower gap dominates. |
| **Offline** | The shim + local LLM can keep working when the platform is unreachable (subject to the shim still being able to reach `localhost:11434`). |

Local LLM routing does **not** replace cloud routing for thinking-mode
tasks, long completions, multi-step planning, or tool use. The gate
below restricts routing to where it actually helps.

---

## Activation gate

A completion turn is routed locally only when **all** of the following
fire:

1. **Executor is a connected `LocalPCShim`** — there's an active
   WebSocket session for this user with the shim daemon. (Cloud E2B
   sandboxes are not local-LLM-capable by construction.)
2. **`local_llm` capability is granted** — `HELLO.capabilities`
   includes the string AND `HELLO_ACK.granted_capabilities` includes
   it (platform-side gate).
3. **`HELLO.local_llm_models` is non-empty** — the shim probed Ollama
   at handshake and found at least one model loaded.
4. **LD flag `LOCAL_LLM_ROUTING` is true for this user** —
   per-user-id rollout. Default off in production.
5. **Per-(mode, tier) routing policy permits it** — see "Routing
   policy" below.
6. **A trusted model match exists** — the requested tier has a
   per-tier preference list; the first match against
   `HELLO.local_llm_models` is picked, and if no model matches the
   tier we skip local routing for this turn.

When any gate fails the platform falls back silently to cloud
completion. The platform-side audit trail records the routing
decision (`local` vs `cloud`) so debug sessions can answer "why
didn't local fire?".

### Routing policy

`copilot.config.local_llm_policy` (new field) — one of:

| Value | Semantics |
|---|---|
| `"never"` | Local LLM routing disabled regardless of the other gates. Default. |
| `"prefer_for_fast"` | Route to local only when `mode == "fast"` (short responses, the "background tab" of the copilot). Thinking-mode tasks go cloud. |
| `"always"` | Route to local whenever the other gates fire, for any mode. |

`prefer_for_fast` is the recommended setting for v1 rollout — it
captures the "small model on user's laptop wins on latency" case
without dragging long-form / planning workloads onto a CPU-bound
inference path.

### Tier model preference

A per-tier list (configurable via LD flag
`LOCAL_LLM_TIER_PREFERENCES`, defaults below) describes which models
to prefer for each tier. The router picks the first model in the
preference list that's also in `HELLO.local_llm_models`.

| Tier | Default preference (in order) |
|---|---|
| `fast` | `llama3.2:3b`, `llama3.2:1b`, `qwen2.5:3b`, `mistral:7b`, `phi3:mini` |
| `default` | `llama3.1:8b`, `qwen2.5:7b`, `mistral:7b` |
| `thinking` | (empty — never route thinking-tier to local in v1) |

Missing tier defaults to "no local routing".

---

## Backends

### v1: Ollama

[Ollama](https://ollama.com) is the v1 (and only v1) backend. It runs
a local HTTP server on `localhost:11434` and exposes an OpenAI-ish
chat-completions API plus a streaming variant. We use it because:

- Single-command install on macOS / Linux / Windows.
- Bundles model weights download + GPU/CPU routing + memory mgmt.
- HTTP API is stable and trivial to mock for tests.
- Models are pulled by name (`ollama pull llama3.2:3b`); the shim
  doesn't need to manage weights.

#### HTTP API the shim speaks

| Op | Method + path | Notes |
|---|---|---|
| List loaded models | `GET /api/tags` | Returns `{models: [{name, ...}]}`. Used at HELLO time to populate `local_llm_models`. |
| Streaming chat | `POST /api/chat` with `{model, messages, stream: true, options: {temperature, top_p, num_predict}}` | Returns a stream of newline-delimited JSON, each `{message: {content: "..."}, done: bool, prompt_eval_count?, eval_count?, total_duration?}`. |
| Non-streaming chat | Same as above with `stream: false` | Returns a single JSON `{message: {content}, done: true, prompt_eval_count, eval_count, total_duration}`. |

Ollama-specific quirks:

- **One-active-request limit.** Ollama serializes requests by default
  (a second `POST /api/chat` blocks until the first finishes). The
  shim tracks active requests via `asyncio.Lock` and rejects the
  second with `LOCAL_LLM_BUSY` rather than queuing — the platform's
  retry layer is the better place for that decision.
- **Model not loaded.** `POST /api/chat` against an unknown model
  returns 404 with `{"error": "model 'foo' not found, ..."}`. The
  shim translates to `MODEL_NOT_AVAILABLE`.
- **Connection refused.** When Ollama isn't running, every call to
  `localhost:11434` returns `ECONNREFUSED`. The shim translates to
  `LOCAL_LLM_FAILED` with `details.backend_error` carrying the raw
  message. At HELLO time, a connection-refused on `/api/tags` causes
  the shim to OMIT the `local_llm` capability and leave
  `local_llm_models` empty — clean degraded mode.

#### Configuration

| Field | Default | Where set |
|---|---|---|
| `config.ollama_url` | `http://localhost:11434` | Shim config TOML / env (`AUTOGPT_SHIM_OLLAMA_URL`). Override to point at a remote Ollama host on the LAN. |
| `config.enable_local_llm` | `false` | Capability gate; must be `true` for the shim to probe Ollama at HELLO time. |

### Future backends

The wire op (`LOCAL_LLM_COMPLETION`) is backend-agnostic; the shim's
dispatch can grow alternative backends behind the same op:

| Backend | When | Notes |
|---|---|---|
| `llama.cpp` server | When users want raw GGUF without Ollama's overhead. | HTTP API similar to Ollama; shim picks based on `config.local_llm_backend`. |
| LM Studio | macOS/Windows GUI users who already have it set up. | Exposes an OpenAI-compatible HTTP API. |
| `mlc-llm` / WebLLM | Browser-side fallback (no shim). | Out of scope for the shim daemon — different routing path entirely. |

For v1 we ship Ollama only. Adding a backend means:
1. New `BackendType` enum value in `config.py`.
2. New handler subclass (mirror `OllamaBackend`).
3. Probe selection logic in `LocalLLMHandler` based on
   `config.local_llm_backend`.

---

## Per-OS notes

### macOS

```sh
brew install ollama
brew services start ollama       # or `ollama serve` in a terminal
ollama pull llama3.2:3b
```

Apple Silicon: Metal accel kicks in automatically. The 3B model uses
~2GB of unified memory; 7B uses ~5GB.

### Linux

```sh
curl -fsSL https://ollama.com/install.sh | sh
systemctl --user enable --now ollama
ollama pull llama3.2:3b
```

NVIDIA: install CUDA + nvidia-container-toolkit; Ollama detects
automatically. AMD: ROCm support is in alpha (as of 2026-06).

### Windows

Use the official installer from
[ollama.com/download](https://ollama.com/download). The installer
registers Ollama as a Windows service that auto-starts. Then:

```pwsh
ollama pull llama3.2:3b
```

Ollama on Windows currently lacks GPU offload for non-NVIDIA cards;
CPU fallback is workable for the 1B/3B models but painful for 7B+.

### Shim model probing

At HELLO time, the shim's `LocalLLMHandler.probe()`:

1. `GET {config.ollama_url}/api/tags` with a 2s timeout.
2. On success: parse `models[].name` and stash the list. Capability
   `local_llm` is included in HELLO; `local_llm_models` carries the
   list.
3. On any failure (timeout, connection refused, 5xx): capability
   `local_llm` is OMITTED from HELLO and `local_llm_models` is empty.
   Log a debug-level message; don't fail the daemon.

The probe runs once per session (at HELLO emission). Model list
changes mid-session (`ollama pull` for a new model) won't be picked up
until the next reconnect — acceptable for v1, since reconnect happens
on every shim restart.

---

## Privacy considerations

Local LLM routing is the foundation of [PRIVACY_MODE.md](PRIVACY_MODE.md);
many of the privacy gains apply even without privacy mode enabled.

### What stays local

- The `messages` array sent in `LOCAL_LLM_COMPLETION` never leaves the
  shim host. The cloud platform sees only the *fact* of the request
  (an audit-log entry) and the *response* if the user's flow surfaces
  it back to the cloud copilot.
- The response (CHUNK + RESPONSE frames) flows back **only** to the
  platform process that initiated the request. The platform-side
  router converts it back into the same SSE event stream the cloud
  models emit, so the rest of the copilot pipeline doesn't need to
  know the difference — but the *transport* never goes back through
  Anthropic/OpenRouter.

### Audit log

Each request is audit-logged as `LOCAL_LLM_COMPLETION` per
[AUDIT_LOG.md](AUDIT_LOG.md) with `details = {model, prompt_chars,
response_chars, duration_ms, finish_reason}`. **Never** logged: the
prompt text, the response text, the conversation context. This matches
the rest of the audit log's posture (sizes and shapes, never bodies).

A user reviewing the audit log can see "the agent made 12
LOCAL_LLM_COMPLETION calls to llama3.2:3b totaling 4.3MB of prompt
text" but not what the prompts said. That's intentional — a user
paranoid enough to care about local routing is paranoid enough not to
want a tamper-evident log of every question they asked their own
model.

### What still leaks (today)

- The *decision* to route locally is visible in the cloud
  platform's request log (it sees "shim was asked, didn't ask
  Anthropic"). That's metadata-level information about user behavior
  but not content.
- Token counts and durations flow back to the platform via the
  `LOCAL_LLM_COMPLETION_RESPONSE` payload. The platform persists
  these in its own analytics tables; this is the same level of
  metadata it'd see for a cloud call.

### Combined with privacy mode

When [PRIVACY_MODE.md](PRIVACY_MODE.md)'s `LOCAL_LLM_TURN` lands (v2),
it adds file-stub references to the prompt construction — same
underlying `LOCAL_LLM_COMPLETION` wire op (`LOCAL_LLM_TURN` is
sugar that builds the messages array from stubs). The privacy gain
compounds: not only does the prompt not transit the cloud, the file
content the prompt is *about* also doesn't.

---

## Errors

See PROTOCOL.md for the full payload format. The codes added for this
feature:

| Code | When | Caller action |
|---|---|---|
| `MODEL_NOT_AVAILABLE` | Requested model not in `HELLO.local_llm_models` or backend returned 404. | Pick a different model from the available list; or fall back to cloud. |
| `LOCAL_LLM_BUSY` | Backend (Ollama) already serving another request. | Retry with backoff (the platform's auto-retry layer does this by default); or fall back to cloud after N retries. |
| `LOCAL_LLM_FAILED` | Backend crashed, HTTP call failed, connection refused. | Surface to user; fall back to cloud for this turn. |

`details.backend_error` carries the raw backend error string on
`LOCAL_LLM_FAILED`. `details.requested_model` /
`details.available_models` carry context on `MODEL_NOT_AVAILABLE`.

---

## Open questions

- **Streaming back-pressure**: today the shim ships chunks as fast as
  the backend produces them. If the WS is slow, chunks queue on the
  asyncio send buffer. v2 should add a slow-consumer signal so the
  shim can pause Ollama's stream rather than buffer indefinitely.
- **GPU contention**: when computer-use is also active (screenshots,
  vision routing), GPU memory becomes the bottleneck. v2 should add
  a probe so the shim can decline LOCAL_LLM if VRAM is tight.
- **Cold-start latency**: first chunk from a freshly-loaded model can
  take 5-10s. v1 doesn't preload; the platform sees the slow first
  chunk and the LD-flag gate is the only knob. v2 could add a "warm"
  signal in HELLO so the router knows which models are ready.

---

## References

- [PROTOCOL.md → Local LLM](PROTOCOL.md#local-llm) — wire op spec.
- [PRIVACY_MODE.md](PRIVACY_MODE.md) — file-stub-aware extension.
- [AUDIT_LOG.md](AUDIT_LOG.md) — per-op `details` shape.
