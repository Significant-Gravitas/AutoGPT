# Running AutoPilot on a self-hosted LLM

> **Important**: This page covers the **AutoPilot chat path** â€” the
> conversational agent on `/copilot`. For the *block-layer* AI Text
> Generator block (used inside agent graphs you build yourself), see
> [Running Ollama with AutoGPT](ollama.md). The two paths read different
> env vars, so configuring one does not configure the other.
>
> Self-hosting only â€” the cloud `agpt.co` deployment routes AutoPilot
> through Anthropic / OpenRouter and ignores the variables below.

This guide makes the AutoPilot chat work **without an Anthropic, OpenAI,
or OpenRouter key** by routing it through any **OpenAI-compatible HTTP
endpoint you control**.

The route is called `local` because it's the typical case, but
`CHAT_LOCAL_BASE_URL` is just a URL â€” every deployment shape below works
equally well:

| Scenario | `CHAT_LOCAL_BASE_URL` |
| --- | --- |
| Ollama on the same Docker host (most common) | `http://192.168.1.42:11434/v1` (LAN IP) |
| Ollama on the same Docker host, via Docker Desktop | `http://host.docker.internal:11434/v1` |
| Ollama on a separate LAN box | `http://ollama.lab.local:11434/v1` |
| Ollama behind an HTTPS reverse proxy on the public internet | `https://ollama.example.com/v1` |
| [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html), [LocalAI](https://localai.io/), [LM Studio](https://lmstudio.ai/), [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) | their respective `/v1` URLs |
| A managed OpenAI-compatible API you don't pay AutoGPT for | its `/v1` URL |

Anything that speaks the OpenAI `/v1/chat/completions` shape â€” including
`tools=[...]` for function calling â€” will work. The rest of this guide
uses Ollama as the running example because it's the easiest, but
substitute your own endpoint anywhere you see `http://...:11434/v1`.

## How it works

AutoPilot's chat runs on one **platform route**, chosen by
`CHAT_PLATFORM_ROUTE`: `openrouter`, `direct_anthropic`, `local`,
`subscription` (this machine's Claude login; self-hosted only) or
`disabled` (no platform model: users chat over the accounts they
connect). Each route reads only its own settings, and the integration
keys (`OPEN_ROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
never configure the chat. With `CHAT_PLATFORM_ROUTE=local`:

| Behaviour | Local route |
| --- | --- |
| Where turns go | `CHAT_LOCAL_BASE_URL`; every turn runs on the same engine as every other route, through AutoPilot's built-in model gateway (`CHAT_LOCAL_ADAPTER` says how it speaks to your server) |
| Key | `CHAT_LOCAL_API_KEY`, only if your server checks one; never `OPENAI_API_KEY` or `OPEN_ROUTER_API_KEY` |
| Models | `CHAT_STANDARD_MODEL` and `CHAT_ADVANCED_MODEL`, both required, sent to the server verbatim |
| Titles, dry runs and other side calls | the standard tier, on the same server |
| Context window | `CHAT_LOCAL_CONTEXT_TOKENS` (required): the window the server has loaded |

On the managed cloud platform (`BEHAVE_AS=cloud`), the route's two
tiers are the *bottom* layer of model resolution: the `copilot-model-routing` flagâ€™s per-user
override â†’ the LLM catalog's routing cell
(`backend/data/llm_registry/catalog.py`) â†’ the tier, with slugs unknown
to the catalog or disabled in it refused at serve time.

**On self-hosted installs â€” any route â€” the catalog's routing cells
are skipped entirely**: they are the cloud deployment's config traveling
in the shipped file, and they never override your `CHAT_STANDARD_MODEL`
/ `CHAT_ADVANCED_MODEL`. Resolution here is the `copilot-model-routing` flag â†’ your tiers;
your settings stay authoritative. See
[Managing LLM Models](contributing/managing-llm-models.md).

Every backend service checks the chat settings when it starts. A
setting from the old format, even one left empty, stops it with a list
naming each such setting and what replaced it, and so does a route
missing a setting it needs. The list names settings, never their
values. Moving an existing local install over:

| Old setting | Now |
| --- | --- |
| `CHAT_USE_LOCAL=true` | `CHAT_PLATFORM_ROUTE=local` |
| `CHAT_BASE_URL` | `CHAT_LOCAL_BASE_URL` |
| `CHAT_API_KEY` | `CHAT_LOCAL_API_KEY`, only if the server checks a key |
| `CHAT_FAST_STANDARD_MODEL` | `CHAT_STANDARD_MODEL` |
| `CHAT_FAST_ADVANCED_MODEL` | `CHAT_ADVANCED_MODEL` (now required; repeat the standard model if you run one) |
| `CHAT_TITLE_MODEL`, `CHAT_SIMULATION_MODEL` | removed: side calls use the standard tier |
| (new) | `CHAT_LOCAL_ADAPTER` and `CHAT_LOCAL_CONTEXT_TOKENS` |

Delete the old lines rather than emptying them. The `--with-ollama`
installers write the new block for you.

**Existing local chats.** Before this version, local turns ran on a
separate, simpler engine. Chats started there keep working: their next
turn runs on the one engine every route uses. It does not resume the old
engine's saved transcript; it rebuilds the conversation from the chat's
stored messages instead, so the first turn of a long chat re-sends its
history and may compact it. That turn saves the chat's new session, which
later turns resume. No message is lost.

## Required environment variables

In `autogpt_platform/backend/.env`:

```bash
# Select the local route
CHAT_PLATFORM_ROUTE=local

# Where the server lives. From inside the docker containers this must
# NOT be 127.0.0.1 / localhost â€” use the host's LAN IP (e.g.
# 192.168.1.42) or, if you've added the directive to compose,
# host.docker.internal:host-gateway.
CHAT_LOCAL_BASE_URL=http://192.168.1.42:11434/v1

# How AutoPilot speaks to the server. openai_chat: an OpenAI-compatible
# server, and CHAT_LOCAL_BASE_URL is its /v1 URL (as above); AutoPilot's
# model gateway translates each turn to /v1/chat/completions.
# anthropic_passthrough: a server that also speaks Anthropic's Messages
# API (llama.cpp's llama-server does), and CHAT_LOCAL_BASE_URL is its root
# (http://192.168.1.42:11434); turns go to its /v1/messages as they are.
CHAT_LOCAL_ADAPTER=openai_chat

# The model's context window as the server has it loaded; see "Context
# window" below. At least 65536.
CHAT_LOCAL_CONTEXT_TOKENS=262144

# The two tiers. Bare model names ONLY â€” provider/model slugs (e.g.
# `anthropic/claude-...`) are passed through verbatim and Ollama can't
# resolve them. Both are required: set the same model twice if you run
# one. Set a bigger advanced model only if you have the VRAM for it.
# See "Picking a model" below.
CHAT_STANDARD_MODEL=hf.co/ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M
CHAT_ADVANCED_MODEL=qwen3:14b-q4_K_M

# Only when the server checks a key (Ollama doesn't).
# CHAT_LOCAL_API_KEY=
```

## Picking a model

The platform's chat loop calls **OpenAI-style tool-calling** on every
turn, streams responses, and ships an ~8 k-token system prompt. Pick a
model that handles all three.

| Tier | Recommended Ollama tag | Why | Footprint |
| --- | --- | --- | --- |
| **Default** | `hf.co/ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M` | Official Ornith GGUF; agentic 9B model with OpenAI-compatible tool calling; 262,144-token native context; reasoning model (the chat UI renders its thinking separately from the answer) | ~5.8 GB model file; allow additional RAM for context and the KV cache |
| **Tight RAM** | `qwen3:4b` | Smaller; native tools; set `think: false` to avoid the unclosed-`<think>` tool-call render bug | ~3-4 GB resident |
| **GPU / advanced** | `qwen3:14b-q4_K_M` | Best tool-selection accuracy in this size class | ~12 GB VRAM |

Pull whichever you choose:

```bash
ollama pull hf.co/ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M
```

## Context window â€” set it on the server, then tell AutoPilot

Ollama defaults `num_ctx` to **4096 tokens regardless of the model's
advertised window** ([ollama/ollama#2714](https://github.com/ollama/ollama/issues/2714)).
That's smaller than AutoPilot's system prompt + tool schemas â€” without a
larger window Ollama only sees the *end* of the instructions and responses
are incoherent or 500 outright.

Set the window **on the server**, via `OLLAMA_CONTEXT_LENGTH`, and put the
same number in `CHAT_LOCAL_CONTEXT_TOKENS`: AutoPilot compacts the
conversation to fit it. AutoPilot takes the window from this setting
alone; it does not read the server's own report (Ollama `/api/ps`,
llama.cpp `/props`). So the setting has to be the real window: a number
too high overflows the model, one too low throws history away.

The default Ornith model has a 262,144-token native window, so the installer
sets `OLLAMA_CONTEXT_LENGTH=262144`, loads the model, and writes the window
Ollama reports into `CHAT_LOCAL_CONTEXT_TOKENS`. It asks when the server
doesn't report one; `--context-tokens=N` (`/context-tokens=N` on Windows)
sets it outright. A large window maximizes available conversation history
but substantially increases KV-cache RAM/VRAM use. Operators using a custom
model or constrained hardware can lower it, but must keep at least 65,536
(64K): AutoPilot's system prompt and tools beside a working conversation
need that much, and the backend refuses to start with a smaller
`CHAT_LOCAL_CONTEXT_TOKENS`.

The installer sets `OLLAMA_CONTEXT_LENGTH` for you. Manual setup per platform:

**Linux** (systemd drop-in):

```ini
# /etc/systemd/system/ollama.service.d/host.conf
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_CONTEXT_LENGTH=262144"
```

Then `sudo systemctl daemon-reload && sudo systemctl restart ollama`.

**macOS** (launchctl, persists across logins for launchd-spawned processes):

```bash
launchctl setenv OLLAMA_HOST 0.0.0.0:11434
launchctl setenv OLLAMA_CONTEXT_LENGTH 262144
# Then restart Ollama â€” either:
brew services restart ollama        # if installed via the brew formula
# â€¦or quit the menu-bar app and relaunch it (the .dmg install)
```

**Windows** (user-scope env vars, persists across reboots):

```powershell
setx OLLAMA_HOST "0.0.0.0:11434"
setx OLLAMA_CONTEXT_LENGTH "262144"
# Then quit Ollama from the system tray and relaunch it
# (setx writes to HKCU but does NOT update already-running processes).
```

Verify on any platform with `ollama ps` (the `CONTEXT` column should
show your value, e.g. 262144). If you change it, set
`CHAT_LOCAL_CONTEXT_TOKENS` to match and restart the backend.

## Networking â€” same host, different host, or remote

The endpoint can be on the same machine, on the LAN, or anywhere
internet-reachable. Pick whichever matches your deployment shape:

### Same host as the AutoGPT containers

How containers reach the host depends on whether you're on Docker
Desktop (macOS / Windows) or native Docker (Linux):

**macOS + Windows (Docker Desktop)** â€” every container already has a
`host.docker.internal` entry pointing at the host. No extra wiring:

```bash
CHAT_LOCAL_BASE_URL=http://host.docker.internal:11434/v1
```

Still set `OLLAMA_HOST=0.0.0.0:11434` so the .app/tray-managed Ollama
accepts the connection from the Desktop network â€” by default it binds
only to `127.0.0.1`.

**Linux (native Docker)** â€” there's no auto-injected
`host.docker.internal`. Pick one:

1. **Use the LAN IP** in `CHAT_LOCAL_BASE_URL` â€” simplest, works everywhere.
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

### Different LAN box (dedicated GPU server, NAS, â€¦)

Set `CHAT_LOCAL_BASE_URL` to the box's hostname or IP:

```bash
CHAT_LOCAL_BASE_URL=http://gpu-rig.lab.local:11434/v1
```

On the Ollama box, set `OLLAMA_HOST=0.0.0.0:11434` so it accepts
non-loopback connections, and either open port 11434 in the firewall
for your AutoGPT host's IP or put both behind a private VPN /
WireGuard mesh.

### Remote / public-internet endpoint

Two approaches, in increasing order of "please do this":

1. **Trusted private network** (Tailscale, WireGuard, ZeroTier,
   corporate VPN). Treat the remote endpoint exactly like a LAN box.
2. **Public HTTPS with auth** â€” terminate TLS at a reverse proxy
   (Caddy, nginx, Cloudflare Tunnel) in front of Ollama / vLLM /
   whatever, and require a bearer token. Set:

   ```bash
   CHAT_LOCAL_BASE_URL=https://ollama.example.com/v1
   CHAT_LOCAL_API_KEY=<the-bearer-the-proxy-checks>
   ```

> **Do not** expose raw Ollama on the public internet. Ollama itself
> performs **no authentication** â€” anyone who can reach `:11434` can
> use (and exhaust) your model. Always front it with a proxy that
> enforces a token.

## Other platform features that use the local route

The `local` route isn't just for AutoPilot chat. The same client
flows to every backend helper that needs an LLM, so a single
`CHAT_PLATFORM_ROUTE=local` install also covers:

- **Dry-run block simulator** â€” when a user clicks "Test" in the agent
  builder, blocks role-play their execution against an LLM rather than
  hitting external APIs. Uses the standard tier (`CHAT_STANDARD_MODEL`).
- **Onboarding business-understanding extraction** â€” the post-signup
  Tally form is extracted into structured suggestions via the LLM.
  Uses the standard tier too.
- **Long-run prompt compression** â€” the chat / agent loop summarizes
  message history when context grows beyond a threshold.
- **Marketplace semantic search** â€” the store generates embeddings for
  agent descriptions to power hybrid (lexical + semantic) ranking.
  Hybrid search degrades gracefully to lexical-only when no embedding
  backend is available.

### Embeddings (marketplace search, agent uploads)

The store's embedding model is overridable via env so deployments with
a compatible backend (vLLM, LiteLLM proxy, Ollama with an embedding
model pulled, Azure OpenAI) can swap models without a code change.
**The replacement model must emit 1536-dim vectors** â€” the pgvector
column is declared `vector(1536)` in `schema.prisma` and inserts with
any other dim hard-fail.

```bash
# Default â€” OpenAI text-embedding-3-small (1536 dim):
STORE_EMBEDDING_MODEL=text-embedding-3-small

# Example: nomic-embed-text on Ollama emits 768 dims natively, so it
# DOES NOT fit the existing schema â€” picking it would break every
# publish + reindex. Use a 1536-dim model instead, e.g.
# text-embedding-ada-002 (OpenAI legacy) or one of the LiteLLM proxy's
# 1536-dim shims. Custom-dim support would need a schema migration
# beyond the scope of this guide.
```

> **pgvector dimension is fixed in the schema, not configurable at
> runtime.** A model that emits a different vector length will succeed
> at the embedding call and fail at every subsequent `INSERT`. If you
> need a different dim, you'll need to fork the schema and migrate
> existing rows â€” it's not a runtime knob.

If you don't configure an embedding backend at all, marketplace
hybrid search auto-degrades to lexical-only (no semantic ranking) â€”
not fatal, just less smart.

## Verifying the wiring

After `docker compose up -d`:

```bash
# 1. The local route is in the live container env
docker exec autogpt_platform-copilot_executor-1 env | grep ^CHAT_
#   CHAT_PLATFORM_ROUTE=local
#   CHAT_LOCAL_BASE_URL=http://192.168.1.42:11434/v1
#   ...

# 2. Send a turn from the UI, then confirm it ran in the log
docker logs autogpt_platform-copilot_executor-1 | grep -E "Using.*service"
#   [CoPilotExecutor|...] Using SDK service

# 3. Confirm Ollama saw the request â€” per platform (POST /v1/chat/completions
#    with CHAT_LOCAL_ADAPTER=openai_chat, /v1/messages with
#    anthropic_passthrough):

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

If `Using SDK service` appears and Ollama logs a 200, the
end-to-end path is working â€” any remaining errors are model / RAM /
quantization concerns rather than config-routing bugs.

## Troubleshooting

**The backend exits at startup, naming chat settings** â€” a setting from
the old format is still set (an empty one counts), or the route is
missing one it needs. Replace or add each setting the list names; see
the table under "How it works" above.

**Frontend shows "The assistant encountered an error"** â€” check the
copilot_executor log for the upstream error. Common causes:
- `model requires more system memory (X GiB) than is available (Y GiB)`
  â†’ free RAM (stop ClamAV, raise VM memory) or pick a smaller model
- `model "..." not found` â†’ `ollama pull <slug>` first
- `connection refused` â†’ containers can't reach the host on `:11434`;
  see "Container â†’ host networking" above

**The server gets no key even though I set `OPENAI_API_KEY`** â€” by
design. The local route sends only `CHAT_LOCAL_API_KEY`, so a cloud key
set for graphiti / embedders never reaches your local server as its
bearer token. Set `CHAT_LOCAL_API_KEY` if the server checks one.

**Title generation fails / returns "Untitled chat"** â€” titles run on
the standard tier (`CHAT_STANDARD_MODEL`) against the same server; check
that model is pulled and answers.

**Slow first response** â€” Ollama loads the model into RAM on the first
request, which can take 5-15 s for 8 B models on CPU. Subsequent
requests are much faster while the model stays resident.

**Every AutoPilot turn takes minutes on CPU** â€” expected on CPU-only
hosts, not a hang. AutoPilot ships an ~8 k-token system prompt and the
model must *prefill* (compute KV-cache state for) every token of that
prompt before the first output token is emitted. On 4 CPU cores an
8 B Q4 model prefills at roughly **3-4 tokens/sec**, so a fresh turn
takes ~35-45 min just to start generating. Title generation (~70-token
prompt) finishes in seconds because there's almost nothing to prefill.
A consumer GPU brings this down to seconds. If you're CPU-only and
just want to validate the install end-to-end, tail the Ollama server
log (see the per-platform commands in "Verifying the wiring" above)
and watch for the `POST /v1/chat/completions` line â€” once it appears
with a 200, prefill finished and the model is generating.

## Dream pass + memory on the local route

The graphiti memory layer and the nightly dream pass both ride the
same self-hosted backend `CHAT_LOCAL_BASE_URL` points at. Three
things you should know:

### Dream pass runs sync-baseline only

The dream pass's batch path (Anthropic batch, OpenAI batch) is
provider-locked and unavailable on local backends.
`CHAT_PLATFORM_ROUTE=local` forces `execution_path="sync_baseline"`
regardless of which API keys might be set elsewhere on the box â€” your
local LLM handles all three phases (consolidate / recombine / sanitize)
on the same endpoint as chat, with the two tiers. Cost-log rows label
`provider="local"` so the admin platform-costs dashboard distinguishes
them from cloud spend.

### Memory uses the chat models by default

On the local route, `GraphitiConfig._apply_local_graphiti_models`
rewrites the cloud OpenAI defaults to local Ollama equivalents:

| Setting | Cloud default | Local default |
|---|---|---|
| `GRAPHITI_LLM_MODEL` | `gpt-4.1-mini` | `hf.co/ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M` |
| `GRAPHITI_RERANKER_MODEL` | `gpt-4.1-mini` | `hf.co/ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M` |
| `GRAPHITI_EMBEDDER_MODEL` | `text-embedding-3-small` | `nomic-embed-text` |

The LLM + reranker reuse the same Ornith 1.5 9B model the `--with-ollama`
installer already pulls for chat, so no extra `ollama pull` is needed
unless you've overridden them. The embedder is a separate model â€” see
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
silently per turn â€” the agent loses memory of the conversation
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
OpenAI / Google upstreams, so on the local and Anthropic routes the
flex client is silently swapped for the regular `OpenAIClient`
(logged at INFO). The weekly community rebuild still runs â€” at full
sync price, which on local Ollama is `$0`.

### Subscription mode caveat

If the chat runs on a Claude Code subscription (`CHAT_PLATFORM_ROUTE=subscription`),
the platform has no model API of its own. The Claude Code OAuth token
authenticates the chat CLI only; per Anthropic's Feb-2026 ToS update, OAuth
tokens **cannot** call the Messages API directly. So the dream pass skips
(reason `no_platform_api`) before any model call, and briefings and teammate
consults go without their model-written parts. An integration key such as
`ANTHROPIC_API_KEY` is never used in its place. Choose an API route
(`openrouter`, `direct_anthropic` or `local`) to run them.

A separate Anthropic Agent-SDK credit pool launches **2026-06-15**
($20 Pro / $100 Max-5x / $200 Max-20x at standard API rates,
one-time opt-in). Once that lands, subscription users will have
the option of routing the dream pass through the Agent SDK instead
of the Messages API â€” coverage tracked as a post-Jun-15 follow-up.
