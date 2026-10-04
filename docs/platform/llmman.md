# Running llmman with AutoGPT

> **Important**: Like Ollama, llmman is only usable when self-hosting the AutoGPT platform. It cannot be used with the cloud-hosted version.

[llmman](https://github.com/llmmanorg/llmman) is a local model runner that serves the Ollama API (alongside OpenAI- and Anthropic-compatible ones) on port **17434**. Because it speaks the same `/api/chat` protocol as Ollama, AutoGPT's existing **Ollama** provider works with it unchanged. Compared with the [Ollama guide](ollama.md), only the port and the launch settings in step 1 differ.

## Prerequisites

1. Complete the [AutoGPT Setup](/platform/getting-started) steps first.
2. Install llmman:

   **Linux/macOS:**
   ```bash
   curl -fsSL https://llmmanorg.github.io/install.sh | sh
   ```

   **Windows (PowerShell):**
   ```powershell
   irm https://llmmanorg.github.io/install.ps1 | iex
   ```

## Setup Steps

### 1. Launch llmman

AutoGPT runs in Docker, so llmman must listen on an address the containers can reach, not its `127.0.0.1` default. Bind it to your host machine's LAN IPv4 address rather than `0.0.0.0`, which would expose it on every network interface. Find the address with `ipconfig` (Windows) or `ip addr show` / `ifconfig` (Linux/macOS); `192.168.0.39` is used as an example below. Set `LLMMAN_HOST` (format `[host][:port]`) and start the server:

**Linux/macOS (Terminal):**
```bash
export LLMMAN_HOST=192.168.0.39:17434
export LLMMAN_AUTH=off
llmman serve
```

**Windows (Command Prompt):**
```cmd
set LLMMAN_HOST=192.168.0.39:17434
set LLMMAN_AUTH=off
llmman serve
```

!!! warning
    AutoGPT's Ollama provider cannot send an API key, so llmman must run with `LLMMAN_AUTH=off`. Without it, llmman refuses to start on an address the network can reach. Anyone who can reach port 17434 can then use your models and manage your model store, so only run this on a trusted network and restrict the port to trusted clients with a firewall or VPN.

In a second terminal, set the same `LLMMAN_HOST` and pull a model. llmman pulls models as OCI artifacts (Docker Hub, GHCR, quay, any registry) or straight from Hugging Face:

```bash
llmman pull gemma4
# or
llmman pull hf.co/unsloth/Qwen3.5-0.8B-GGUF
```

Check what is available with:
```bash
curl http://192.168.0.39:17434/api/tags
```

### 2. Allow the backend to reach llmman

The Ollama host you enter in a block is checked against an SSRF allowlist. Private/LAN addresses are rejected unless they match `OLLAMA_HOST` in the backend environment, so add the same host IP and llmman's port to `autogpt_platform/backend/.env`:

```bash
OLLAMA_HOST=192.168.0.39:17434
```

Then start (or restart) the platform:

```bash
cd autogpt_platform
docker compose up -d --build
```

### 3. Using llmman with AutoGPT

1. Open [http://localhost:3000/build](http://localhost:3000/build) and add an AI Text Generator block (any AI LLM block works).
2. **API Key**: enter any value (e.g. `not-needed`). llmman does not require authentication here.
3. **LLM Model**: pick an **Ollama** model. The name AutoGPT sends must match a model llmman has pulled, so either pull a model under one of the built-in Ollama slugs (`llama3.2`, `llama3`, `llama3.1:405b`, `dolphin-mistral:latest`) or add your own (see below).
4. **Ollama Host**: enter the same value you put in `OLLAMA_HOST`, e.g. `192.168.0.39:17434`.
5. Add a prompt, save the graph, and run it.

## Using other models

To expose a model such as `gemma4` or `hf.co/unsloth/Qwen3.5-0.8B-GGUF` in the dropdown, add a catalog entry with `provider="ollama"` and a matching `LLMModel` name, exactly as described in [Add Custom Models](ollama.md#add-custom-models-advanced). The slug must be the name llmman serves it under (as listed by `/api/tags`).

## AutoPilot (OpenAI-compatible path)

The [AutoPilot self-hosted LLM guide](copilot-local-llm.md) uses the OpenAI-compatible `/v1` endpoint rather than the Ollama API. llmman serves `/v1/chat/completions` (with tools), so the same guide applies with `CHAT_BASE_URL=http://<host-ip>:17434/v1`. Ollama-only settings in that guide need the llmman equivalent:

- Set the context window with `LLMMAN_CONTEXT_LENGTH` on the llmman server instead of `OLLAMA_CONTEXT_LENGTH`. It defaults to 262144 for `llama-server`, capped to the model's trained context, so lower it on memory-constrained hardware (AutoPilot needs at least 24k).
- Use `llmman pull` and `llmman ps` in place of `ollama pull` and `ollama ps`.
- `/v1/embeddings` is not available when llmman runs on macOS's `mlx_lm.server` backend. Embeddings must also be 1536-dimensional, as that guide describes.

## Troubleshooting

- **Connection refused**: use your host machine's IP rather than `localhost`/`127.0.0.1`, and make sure llmman was started with `LLMMAN_HOST` set to that IP and port. Verify with `curl http://192.168.0.39:17434/api/tags` (your IP) on the host.
- **llmman exits at startup asking for an API key**: it is bound to a network-reachable address without `LLMMAN_AUTH=off`. See step 1.
- **Host rejected / SSRF error**: the value in the "Ollama Host" field must match `OLLAMA_HOST` in `backend/.env` (hostname and port).
- **Model not found**: the selected model must be pulled in llmman under exactly that name.
