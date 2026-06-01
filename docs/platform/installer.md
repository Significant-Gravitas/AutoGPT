# AutoGPT Platform Installer

The AutoGPT Platform provides easy-to-use installers to help you quickly set up the platform on your system. This page covers how to use the installer scripts for both Linux/macOS and Windows.

## What the Installer Does

The **bootstrap installer** (`install.sh` / `install.ps1`) is a zero-prerequisite, one-line entry point. It:

1. **Pre-flight check** — verifies your machine can actually run AutoGPT (OS/version, CPU virtualization, RAM, free disk, admin rights) and **tells you up front, with a fix,** if something is a hard blocker (e.g. virtualization disabled in BIOS) — *before* installing anything.
2. **Installs prerequisites** — git and Docker if they're missing (Docker Engine on Linux, Docker Desktop on Windows/macOS). Already-present tools are detected and skipped.
3. **Fetches the repo** at the version you choose — the latest release (default), the `dev` branch, or a custom branch/tag.
4. **Hands off** to `setup-autogpt.{sh,bat}`, which builds the Docker stack, optionally wires a local LLM, and starts everything.

> Already have git + Docker and just want the platform setup? Run `setup-autogpt.{sh,bat}` directly (see [Manual Installation](#manual-installation)).

## Prerequisites

**None required up front.** The bootstrap installer checks for and installs git and Docker for you, and runs a pre-flight check first so you know whether your hardware/OS can run them. The only hard requirements it verifies:

- **CPU virtualization** available/enabled — Docker needs it on Windows/macOS (WSL2 / Apple Hypervisor). On Linux, Docker is native and this isn't required.
- **~25 GB free disk** and **8 GB+ RAM** for the full stack.
- **Admin / sudo** to install Docker and enable WSL2 (Windows).

## Quick One-Liner Installation

These commands install everything from scratch — prerequisites included.

### Linux / macOS

```bash
curl -fsSL https://setup.agpt.co/install.sh | bash
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -Command "iwr https://setup.agpt.co/install.ps1 -OutFile install.ps1; ./install.ps1"
```

### Options

Both installers share the same flags:

| Goal | Linux/macOS | Windows |
|------|-------------|---------|
| Latest release (default) | *(no flag)* | *(no flag)* |
| `dev` branch | `--dev` | `-Dev` |
| Specific branch | `--branch=NAME` | `-Branch NAME` |
| Specific release | `--release=TAG` | `-Release TAG` |
| Local LLM, no cloud keys | `--with-ollama` | `-WithOllama` |
| Just check my machine | `--preflight-only` | `-PreflightOnly` |
| Custom install dir | `--dir=PATH` | `-Dir PATH` |

Example (dev branch + local LLM):

```bash
curl -fsSL https://setup.agpt.co/install.sh | bash -s -- --dev --with-ollama
```

## Manual Installation

If you prefer, you can manually download and run the installer scripts:

- **Linux/macOS:** `setup-autogpt.sh`
- **Windows:** `setup-autogpt.bat`

These scripts are located in the `autogpt_platform/installer/` directory.

## Running fully offline with a local LLM (Ollama)

Both installer scripts accept an opt-in flag that installs
[Ollama](https://ollama.com), pulls a default chat model, and wires
`backend/.env` so AutoPilot runs **without any cloud API keys**. This
is useful for air-gapped or privacy-sensitive deployments — see
[Running AutoPilot on a self-hosted LLM](copilot-local-llm.md) for the
full reference.

### Linux / macOS

```bash
cd autogpt_platform/installer
./setup-autogpt.sh --with-ollama
# Optional overrides:
#   --ollama-model=qwen3:14b-instruct-q4_K_M
#   --ollama-host=http://gpu-rig.lab:11434   # use an existing Ollama
```

### Windows

```cmd
cd autogpt_platform\installer
setup-autogpt.bat /with-ollama
REM Optional overrides:
REM   /ollama-model=qwen3:14b-instruct-q4_K_M
REM   /ollama-host=http://gpu-rig.lab:11434
```

The installer:

1. Installs Ollama (skipped if already present, or if `--ollama-host` points at an existing one).
2. Configures `OLLAMA_HOST=0.0.0.0:11434` + `OLLAMA_CONTEXT_LENGTH=32768` so containers can reach it and so AutoPilot's ~8 k system prompt isn't truncated by Ollama's 4 k default.
3. Pulls the chat model (default `llama3.1:8b-instruct-q4_K_M`).
4. Appends a marker-bounded block to `autogpt_platform/backend/.env` with `CHAT_USE_LOCAL=true` plus the `CHAT_BASE_URL` / `CHAT_API_KEY` / `CHAT_*_MODEL` overrides.

Re-running with `--with-ollama` is idempotent — the wiring block is rewritten in place.

## After Installation

Once the installation is complete:
- The backend services will be running in Docker containers
- The frontend application will be available at http://localhost:3000

## Stopping the Services

To stop the services, press Ctrl+C in the terminal where the frontend is running, then run:

```bash
cd AutoGPT/autogpt_platform
docker compose down
```

## Troubleshooting

If you encounter any issues during installation:

1. Make sure all prerequisites are correctly installed
2. Check that Docker is running
3. Ensure you have a stable internet connection
4. Verify you have sufficient permissions to create directories and run Docker 