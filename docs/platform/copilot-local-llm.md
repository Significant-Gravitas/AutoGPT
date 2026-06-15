# Running AutoPilot on a self-hosted LLM

> **Important**: This page covers the **AutoPilot chat path** — the
> conversational agent on `/copilot`. For the *block-layer* AI Text
> Generator block (used inside agent graphs you build yourself), see
> [Running Ollama with AutoGPT](ollama.md). The two paths read different
> env vars, so configuring one does not configure the other.
>
> Self-hosting only — the cloud `agpt.co` deployment routes AutoPilot
> through Anthropic / OpenRouter and ignores the variables below.

This guide makes the AutoPilot chat work **without an Anthropic, OpenAI,
or OpenRouter key** by routing it through any **OpenAI-compatible HTTP
endpoint you control**.

The transport is called `local` because it's the typical case, but
``CHAT_BASE_URL`` is just a URL — every deployment shape below works
equally well:

| Scenario | `CHAT_BASE_URL` |
| --- | --- |
| Ollama on the same Docker host (most common) | `http://192.168.1.42:11434/v1` (LAN IP) |
| Ollama on the same Docker host, via Docker Desktop | `http://host.docker.internal:11434/v1` |
| Ollama on a separate LAN box | `http://ollama.lab.local:11434/v1` |
| Ollama behind an HTTPS reverse proxy on the public internet | `https://ollama.example.com/v1` |
| [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html), [LocalAI](https://localai.io/), [LM Studio](https://lmstudio.ai/), [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) | their respective `/v1` URLs |
| A managed OpenAI-compatible API you don't pay AutoGPT for | its `/v1` URL |

Anything that speaks the OpenAI `/v1/chat/completions` shape — including
`tools=[...]` for function calling — will work. The rest of this guide
uses Ollama as the running example because it's the easiest, but
substitute your own endpoint anywhere you see `http://...:11434/v1`.

## How it works

AutoPilot's `ChatConfig` (`backend/backend/copilot/config.py`) recognises
four chat transports. When `CHAT_USE_LOCAL=true`:

| Transport behaviour | Local |
| --- | --- |
| Routes the baseline (fast) path to `CHAT_BASE_URL` over OpenAI-compatible HTTP | ✅ |
| Supports the SDK / extended-thinking path (Claude Agent SDK) | ❌ — auto-downgrades to fast |
| `api_key` falls back to `OPEN_ROUTER_API_KEY` / `OPENAI_API_KEY` if `CHAT_API_KEY` is unset | ❌ — explicit `CHAT_API_KEY` only |
| Aux + advanced models (`title_model`, `simulation_model`, `fast_advanced_model`) inherit `fast_standard_model` if left at a cloud default | ✅ |
| Allows non-`anthropic/*` SDK model slugs (vendor validator skipped) | ✅ |

The downgrade is logged at WARNING when an `extended_thinking` request
arrives — there is no 500. The frontend toggle should already be hidden
because the `CHAT_MODE_OPTION` LaunchDarkly flag defaults off in
self-hosted deployments.

## Required environment variables

In `autogpt_platform/backend/.env`:

```bash
# Turn on the local transport
CHAT_USE_LOCAL=true

# Where the OpenAI-compatible endpoint lives. From inside the docker
# containers this must NOT be 127.0.0.1 / localhost — use the host's LAN
# IP (e.g. 192.168.1.42) or, if you've added the directive to compose,
# host.docker.internal:host-gateway.
CHAT_BASE_URL=http://192.168.1.42:11434/v1

# Any non-empty string — Ollama doesn't validate it. The local transport
# deliberately does NOT fall back to OPENAI_API_KEY (which is usually
# present for graphiti / embedders), so this must be set explicitly.
CHAT_API_KEY=ollama

# The chat model. Bare model names ONLY — provider/model slugs (e.g.
# `anthropic/claude-...`) are passed through verbatim and Ollama can't
# resolve them. See "Picking a model" below.
CHAT_FAST_STANDARD_MODEL=hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M

# Optional — override for the advanced tier. If you leave it out, the
# local transport derives title_model, simulation_model, AND
# CHAT_FAST_ADVANCED_MODEL from CHAT_FAST_STANDARD_MODEL automatically
# (see _apply_local_aux_models in backend/backend/copilot/config.py),
# so the advanced toggle never sends a cloud-only slug to Ollama. Set
# it explicitly only if you want a bigger model for the advanced tier
# and have the VRAM for it.
CHAT_FAST_ADVANCED_MODEL=qwen3:14b-q4_K_M
```

## Picking a model

The platform's chat loop calls **OpenAI-style tool-calling** on every
turn, streams responses, and ships an ~8 k-token system prompt. Pick a
model that handles all three.

| Tier | Recommended Ollama tag | Why | Footprint |
| --- | --- | --- | --- |
| **Default** | `hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M` | Unsloth-recommended; solid OpenAI-shim tool-calling at 4B; 256 k native context; reasoning model (the chat UI renders its thinking separately from the answer) | ~3.4 GB resident; runs on a 16 GB laptop, GPU-accelerated |
| **Tight RAM** | `qwen3:4b` | Smaller; native tools; set `think: false` to avoid the unclosed-`<think>` tool-call render bug | ~3-4 GB resident |
| **GPU / advanced** | `qwen3:14b-q4_K_M` | Best tool-selection accuracy in this size class | ~12 GB VRAM |

Pull whichever you choose:

```bash
ollama pull hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M
```

## Context window — set it once, on the backend

Ollama defaults `num_ctx` to **4096 tokens regardless of the model's
advertised window** ([ollama/ollama#2714](https://github.com/ollama/ollama/issues/2714)).
That's smaller than AutoPilot's system prompt + tool schemas — without a
larger window Ollama only sees the *end* of the instructions and responses
are incoherent or 500 outright.

Set the window **once, on the server**, via `OLLAMA_CONTEXT_LENGTH`. There is
**no AutoGPT-side context config** to keep in sync: AutoPilot reads the
backend's *actual* loaded window back at runtime — Ollama `/api/ps`, llama.cpp
`/props`, vLLM / LM Studio `/v1/models` — and compacts the conversation under
it. Backends that don't report a window (LiteLLM proxy, Jan,
text-generation-webui) fall back to assuming 32k.

Use **at least 24k** (32768 recommended): below that, the system prompt +
tools leave almost no room for conversation and AutoPilot logs a warning.

The installer sets `OLLAMA_CONTEXT_LENGTH` for you. Manual setup per platform:

**Linux** (systemd drop-in):

```ini
# /etc/systemd/system/ollama.service.d/host.conf
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_CONTEXT_LENGTH=32768"
```

Then `sudo systemctl daemon-reload && sudo systemctl restart ollama`.

**macOS** (launchctl, persists across logins for launchd-spawned processes):

```bash
launchctl setenv OLLAMA_HOST 0.0.0.0:11434
launchctl setenv OLLAMA_CONTEXT_LENGTH 32768
# Then restart Ollama — either:
brew services restart ollama        # if installed via the brew formula
# …or quit the menu-bar app and relaunch it (the .dmg install)
```

**Windows** (user-scope env vars, persists across reboots):

```powershell
setx OLLAMA_HOST "0.0.0.0:11434"
setx OLLAMA_CONTEXT_LENGTH "32768"
# Then quit Ollama from the system tray and relaunch it
# (setx writes to HKCU but does NOT update already-running processes).
```

Verify on any platform with `ollama ps` (the `CONTEXT` column should
show your value, e.g. 32768). If you change it, AutoPilot picks up the
new window automatically on the next turn — nothing else to update.

## Networking — same host, different host, or remote

The endpoint can be on the same machine, on the LAN, or anywhere
internet-reachable. Pick whichever matches your deployment shape:

### Same host as the AutoGPT containers

How containers reach the host depends on whether you're on Docker
Desktop (macOS / Windows) or native Docker (Linux):

**macOS + Windows (Docker Desktop)** — every container already has a
`host.docker.internal` entry pointing at the host. No extra wiring:

```bash
CHAT_BASE_URL=http://host.docker.internal:11434/v1
```

Still set `OLLAMA_HOST=0.0.0.0:11434` so the .app/tray-managed Ollama
accepts the connection from the Desktop network — by default it binds
only to `127.0.0.1`.

**Linux (native Docker)** — there's no auto-injected
`host.docker.internal`. Pick one:

1. **Use the LAN IP** in `CHAT_BASE_URL` — simplest, works everywhere.
2. **Bind Ollama to all interfaces:** `OLLAMA_HOST=0.0.0.0:11434` (set
   in the systemd unit or a drop-in), so containers reach it via the
   bridge gateway.
3. **Add `extra_hosts: ["host.docker.internal:host-gateway"]`** to the
   chat services in `autogpt_platform/docker-compose.yml`.

The bundled installer does these for you on a fresh box:

| Platform | Command |
| --- | --- |
| Linux | `installer/setup-autogpt.sh --with-ollama` |
| macOS | `installer/setup-autogpt.sh --with-ollama` |
| Windows | `installer\setup-autogpt.bat /with-ollama` |

### Different LAN box (dedicated GPU server, NAS, …)

Set `CHAT_BASE_URL` to the box's hostname or IP:

```bash
CHAT_BASE_URL=http://gpu-rig.lab.local:11434/v1
```

On the Ollama box, set `OLLAMA_HOST=0.0.0.0:11434` so it accepts
non-loopback connections, and either open port 11434 in the firewall
for your AutoGPT host's IP or put both behind a private VPN /
WireGuard mesh.

### Remote / public-internet endpoint

Two approaches, in increasing order of "please do this":

1. **Trusted private network** (Tailscale, WireGuard, ZeroTier,
   corporate VPN). Treat the remote endpoint exactly like a LAN box.
2. **Public HTTPS with auth** — terminate TLS at a reverse proxy
   (Caddy, nginx, Cloudflare Tunnel) in front of Ollama / vLLM /
   whatever, and require a bearer token. Set:

   ```bash
   CHAT_BASE_URL=https://ollama.example.com/v1
   CHAT_API_KEY=<the-bearer-the-proxy-checks>
   ```

> **Do not** expose raw Ollama on the public internet. Ollama itself
> performs **no authentication** — anyone who can reach `:11434` can
> use (and exhaust) your model. Always front it with a proxy that
> enforces a token.

## Other platform features that use the local transport

The `local` transport isn't just for AutoPilot chat. The same client
flows to every backend helper that needs an LLM, so a single
`CHAT_USE_LOCAL=true` install also covers:

- **Dry-run block simulator** — when a user clicks "Test" in the agent
  builder, blocks role-play their execution against an LLM rather than
  hitting external APIs. Uses `ChatConfig.simulation_model`
  (auto-derived to `fast_standard_model` under local).
- **Onboarding business-understanding extraction** — the post-signup
  Tally form is extracted into structured suggestions via the LLM.
  Uses `ChatConfig.title_model`.
- **Long-run prompt compression** — the chat / agent loop summarizes
  message history when context grows beyond a threshold.
- **Marketplace semantic search** — the store generates embeddings for
  agent descriptions to power hybrid (lexical + semantic) ranking.
  Hybrid search degrades gracefully to lexical-only when no embedding
  backend is available.

### Embeddings (marketplace search, agent uploads)

The store's embedding model is overridable via env so deployments with
a compatible backend (vLLM, LiteLLM proxy, Ollama with an embedding
model pulled, Azure OpenAI) can swap models without a code change.
**The replacement model must emit 1536-dim vectors** — the pgvector
column is declared `vector(1536)` in `schema.prisma` and inserts with
any other dim hard-fail.

```bash
# Default — OpenAI text-embedding-3-small (1536 dim):
STORE_EMBEDDING_MODEL=text-embedding-3-small

# Example: nomic-embed-text on Ollama emits 768 dims natively, so it
# DOES NOT fit the existing schema — picking it would break every
# publish + reindex. Use a 1536-dim model instead, e.g.
# text-embedding-ada-002 (OpenAI legacy) or one of the LiteLLM proxy's
# 1536-dim shims. Custom-dim support would need a schema migration
# beyond the scope of this guide.
```

> **pgvector dimension is fixed in the schema, not configurable at
> runtime.** A model that emits a different vector length will succeed
> at the embedding call and fail at every subsequent `INSERT`. If you
> need a different dim, you'll need to fork the schema and migrate
> existing rows — it's not a runtime knob.

If you don't configure an embedding backend at all, marketplace
hybrid search auto-degrades to lexical-only (no semantic ranking) —
not fatal, just less smart.

## Verifying the wiring

After `docker compose up -d`:

```bash
# 1. CHAT_USE_LOCAL is in the live container env
docker exec autogpt_platform-copilot_executor-1 env | grep ^CHAT_
#   CHAT_USE_LOCAL=true
#   CHAT_BASE_URL=http://192.168.1.42:11434/v1
#   ...

# 2. Send a turn from the UI, then confirm baseline routing in the log
docker logs autogpt_platform-copilot_executor-1 | grep -E "Using.*service"
#   [CoPilotExecutor|...] Using baseline service (mode=default)

# 3. Confirm Ollama saw the request — per platform:

# Linux (systemd-managed Ollama):
journalctl -u ollama --since "1 minute ago" | grep "POST"
#   [GIN] ... | 200 | 7.5s |  ... | POST "/v1/chat/completions"

# macOS (brew formula):
tail -F "$(brew --prefix)/var/log/ollama.log" | grep "POST"
# macOS (.app from ollama.com): logs live in ~/.ollama/logs/server.log
tail -F ~/.ollama/logs/server.log | grep "POST"

# Windows: the Ollama tray app writes to %LOCALAPPDATA%\Ollama\server.log
powershell -Command "Get-Content $env:LOCALAPPDATA\Ollama\server.log -Wait | Select-String POST"
```

If `Using baseline service` appears and Ollama logs a 200, the
end-to-end path is working — any remaining errors are model / RAM /
quantization concerns rather than config-routing bugs.

## Troubleshooting

**Frontend shows "The assistant encountered an error"** — check the
copilot_executor log for the upstream error. Common causes:
- `model requires more system memory (X GiB) than is available (Y GiB)`
  → free RAM (stop ClamAV, raise VM memory) or pick a smaller model
- `model "..." not found` → `ollama pull <slug>` first
- `connection refused` → containers can't reach the host on `:11434`;
  see "Container → host networking" above

**`api_key` is `None` even though I set `OPENAI_API_KEY`** — by design.
The local transport requires an explicit `CHAT_API_KEY` so a stray cloud
key set for graphiti / embedders doesn't silently bind to your local
backend as the bearer token.

**Title generation fails / returns "Untitled chat"** — `title_model`
should auto-inherit `fast_standard_model` under the local transport. If
you've explicitly set `CHAT_TITLE_MODEL=openai/gpt-4o-mini` somewhere,
remove it.

**Slow first response** — Ollama loads the model into RAM on the first
request, which can take 5-15 s for 8 B models on CPU. Subsequent
requests are much faster while the model stays resident.

**Every AutoPilot turn takes minutes on CPU** — expected on CPU-only
hosts, not a hang. AutoPilot ships an ~8 k-token system prompt and the
model must *prefill* (compute KV-cache state for) every token of that
prompt before the first output token is emitted. On 4 CPU cores an
8 B Q4 model prefills at roughly **3-4 tokens/sec**, so a fresh turn
takes ~35-45 min just to start generating. Title generation (~70-token
prompt) finishes in seconds because there's almost nothing to prefill.
A consumer GPU brings this down to seconds. If you're CPU-only and
just want to validate the install end-to-end, tail the Ollama server
log (see the per-platform commands in "Verifying the wiring" above)
and watch for the `POST /v1/chat/completions` line — once it appears
with a 200, prefill finished and the model is generating.

## Dream pass + memory under local transport

The graphiti memory layer and the nightly dream pass both ride the
same self-hosted backend `CHAT_USE_LOCAL=true` points at. Three
things you should know:

### Dream pass runs sync-baseline only

The dream pass's batch path (Anthropic batch, OpenAI batch) is
provider-locked and unavailable on local backends. `CHAT_USE_LOCAL=true`
forces `execution_path="sync_baseline"` regardless of which API keys
might be set elsewhere on the box — your local LLM handles all three
phases (consolidate / recombine / sanitize) on the same endpoint as
chat. Cost-log rows label `provider="ollama"` so the admin
platform-costs dashboard distinguishes them from cloud spend.

### Memory uses the chat models by default

When `CHAT_USE_LOCAL=true`, `GraphitiConfig._apply_local_graphiti_models`
rewrites the cloud OpenAI defaults to local Ollama equivalents:

| Setting | Cloud default | Local default |
|---|---|---|
| `GRAPHITI_LLM_MODEL` | `gpt-4.1-mini` | `hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M` |
| `GRAPHITI_RERANKER_MODEL` | `gpt-4.1-nano` | `hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M` |
| `GRAPHITI_EMBEDDER_MODEL` | `text-embedding-3-small` | `nomic-embed-text` |

The LLM + reranker reuse the same Qwen 3.5 4B model the `--with-ollama`
installer already pulls for chat, so no extra `ollama pull` is needed
unless you've overridden them. The embedder is a separate model — see
the next section.

You can pin your own slugs at any time by setting the matching
`GRAPHITI_*_MODEL` env var; the validator only touches slots still at
their cloud default. A custom slug (`qwen3:8b`, `hf.co/...`,
`my-registry.io/model:tag`) passes through untouched.

### Embeddings require an embedding model pulled into Ollama

Ollama doesn't ship an embedding model in its default model set, so
graphiti's per-turn entity extraction will 404 on `/v1/embeddings`
until you pull one:

```bash
ollama pull nomic-embed-text
```

Without it, chat still works, but `graphiti.add_episode(...)` fails
silently per turn — the agent loses memory of the conversation
between sessions. With it pulled, graphiti round-trips
embeddings against the same Ollama endpoint as the LLM, and
warm-context retrieval works end-to-end on the local stack.

If you'd rather use a different embedding model (e.g.
`mxbai-embed-large` for higher recall at higher disk cost), pull
that and set `GRAPHITI_EMBEDDER_MODEL=<slug>` to override the
local default.

### Community rebuild stays on sync tier

`graphiti_config.community_rebuild_use_flex_tier=True` (the default)
is treated as a *request*, not a guarantee. OpenAI's flex tier only
delivers the ~50% discount through OpenRouter's pass-through to
OpenAI / Google upstreams, so on local + Anthropic transports the
flex client is silently swapped for the regular `OpenAIClient`
(logged at INFO). The weekly community rebuild still runs — at full
sync price, which on local Ollama is `$0`.

### Subscription mode caveat

If you also use Claude Code subscription (`CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=true`)
for the chat path, the dream pass needs a separate `ANTHROPIC_API_KEY`
set in the environment. The Claude Code OAuth token authenticates the
chat CLI only; per Anthropic's Feb-2026 ToS update, OAuth tokens
**cannot** call the Messages API directly. Without `ANTHROPIC_API_KEY`,
the dream pass writes an `errored` JobStatus with a friendly hint
pointing you at this section.

A separate Anthropic Agent-SDK credit pool launches **2026-06-15**
($20 Pro / $100 Max-5x / $200 Max-20x at standard API rates,
one-time opt-in). Once that lands, subscription users will have
the option of routing the dream pass through the Agent SDK instead
of the Messages API — coverage tracked as a post-Jun-15 follow-up.
