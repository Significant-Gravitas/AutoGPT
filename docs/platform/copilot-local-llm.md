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
| Auxiliary models (`title_model`, `simulation_model`) inherit `fast_standard_model` if left at default | ✅ |
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
CHAT_FAST_STANDARD_MODEL=llama3.1:8b-instruct-q4_K_M

# Optional — overrides for advanced tier and aux models. If you leave
# them out, the local transport derives title_model and simulation_model
# from CHAT_FAST_STANDARD_MODEL automatically; CHAT_FAST_ADVANCED_MODEL
# stays at its cloud default and you should set it to the same Ollama
# slug (or a bigger one if you have the VRAM).
CHAT_FAST_ADVANCED_MODEL=llama3.1:8b-instruct-q4_K_M
```

## Picking a model

The platform's chat loop calls **OpenAI-style tool-calling** on every
turn, streams responses, and ships an ~8 k-token system prompt. Pick a
model that handles all three.

| Tier | Recommended Ollama tag | Why | Footprint |
| --- | --- | --- | --- |
| **Default** | `llama3.1:8b-instruct-q4_K_M` | Battle-tested OpenAI-shim tool-calling; 128 k advertised context; no `<think>` tag artifacts | ~6 GB resident, CPU-friendly |
| **Tight RAM** | `qwen3:4b` | Smaller; native tools; set `think: false` to avoid the unclosed-`<think>` tool-call render bug | ~3-4 GB resident |
| **GPU / advanced** | `qwen3:14b-instruct-q4_K_M` | Best tool-selection accuracy in this size class | ~12 GB VRAM |

Pull whichever you choose:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

## The `num_ctx` gotcha

Ollama defaults `num_ctx` to **4096 tokens regardless of the model's
advertised window** ([ollama/ollama#2714](https://github.com/ollama/ollama/issues/2714)).
That's smaller than AutoPilot's system prompt — without a ctx override
Ollama only sees the *end* of the instructions, the response is
incoherent or 500s outright.

**Important:** Ollama's OpenAI-compatibility endpoint (`/v1/chat/completions`)
**does NOT** honor an `options.num_ctx` field in the request body — only
the native `/api/chat` does. The OpenAI client inside the AutoPilot
backend talks to the `/v1` shim, so the chat-side `CHAT_LOCAL_NUM_CTX`
config is forwarded **for OpenAI-compatible backends that DO honor it
in the request body** (vLLM, LM Studio, LiteLLM proxy …).

For Ollama specifically, set the context at the **server** via the
`OLLAMA_CONTEXT_LENGTH` env var on the systemd unit — this is what
the bundled installer's `--with-ollama` flag does:

```ini
# /etc/systemd/system/ollama.service.d/host.conf
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_CONTEXT_LENGTH=32768"
```

Then `sudo systemctl daemon-reload && sudo systemctl restart ollama`
and verify with `ollama ps` (the `CONTEXT` column should show 32768).

## Networking — same host, different host, or remote

The endpoint can be on the same machine, on the LAN, or anywhere
internet-reachable. Pick whichever matches your deployment shape:

### Same host as the AutoGPT containers

On Linux, containers can't reach the host via `localhost` /
`host.docker.internal` unless you wire it explicitly. Either:

1. **Use the LAN IP** in `CHAT_BASE_URL` — simplest, works everywhere.
2. **Bind Ollama to all interfaces:** `OLLAMA_HOST=0.0.0.0:11434` (set
   in the systemd unit or a drop-in), so containers reach it via the
   bridge gateway.
3. **Add `extra_hosts: ["host.docker.internal:host-gateway"]`** to the
   chat services in `autogpt_platform/docker-compose.yml` — works on
   Linux + already automatic on Docker Desktop.

The bundled `installer/setup-autogpt.sh --with-ollama` flag does
options (1) + (2) for you on a fresh Linux box.

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
model pulled, Azure OpenAI) can swap models without a code change:

```bash
# Default (matches the historical OpenAI text-embedding-3-small column shape):
STORE_EMBEDDING_MODEL=text-embedding-3-small
STORE_EMBEDDING_DIM=1536

# Example: nomic-embed-text on Ollama (smallest and fastest decent embedder):
ollama pull nomic-embed-text
# then in backend/.env:
STORE_EMBEDDING_MODEL=nomic-embed-text
STORE_EMBEDDING_DIM=768
```

> **pgvector dimension is fixed at column-create time.** If you change
> `STORE_EMBEDDING_DIM` after content has already been embedded, the new
> writes will fail with a shape mismatch — you'll need to re-create the
> embedding column or drop existing rows. Pick a dim at install time
> and keep it.

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

# 3. Confirm Ollama saw the request
journalctl -u ollama --since "1 minute ago" | grep "POST"
#   [GIN] ... | 200 | 7.5s |  ... | POST "/v1/chat/completions"
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
hosts, not a hang. AutoPilot ships a ~3 k-token system prompt and the
model must *prefill* (compute KV-cache state for) every token of that
prompt before the first output token is emitted. On 4 CPU cores an
8 B Q4 model prefills at roughly **3-4 tokens/sec**, so a fresh turn
takes ~10-15 min just to start generating. Title generation (~70-token
prompt) finishes in seconds because there's almost nothing to prefill.
A consumer GPU brings this down to seconds. If you're CPU-only and
just want to validate the install end-to-end, watch
`journalctl -u ollama -f` for the `POST /v1/chat/completions` line —
once it appears with a 200, prefill finished and the model is
generating.
