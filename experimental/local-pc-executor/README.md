# ⚠️ EXPERIMENTAL — AutoGPT Local PC Executor

> **DANGER: Untested, pre-alpha, experimental software.**
> Do not run on any machine you care about. Do not run as root.
> This *maybe* gives the AutoGPT hosted platform the ability to execute commands on your computer.

---

## What Is This?

A daemon you install on your local machine that connects it to the [AutoGPT hosted platform](https://platform.autogpt.net) as an execution backend — instead of an E2B cloud sandbox.

Once connected, AutoGPT can:
- Read and write files on your filesystem (jailed to a configurable root)
- Execute shell commands
- *(Optional)* Take screenshots and control mouse/keyboard via Claude's computer use API
- *(Optional)* Access local hardware (serial, USB, GPIO)
- *(Optional)* Route LLM inference to a local Ollama instance

## Platform Binding Layer

The platform-side code (WebSocket route, `LocalPCShim`, `ShimConnectionManager`, config hooks) lives in the AutoGPT monorepo:
👉 https://github.com/Significant-Gravitas/AutoGPT/pull/13050

## Current Status

| Component | Status |
|-----------|--------|
| Protocol spec | 🟡 Draft |
| Shim daemon | 🔴 Skeleton / stubs only |
| OAuth flow | 🔴 Spec only |
| Computer use | 🔴 Spec only |
| Hardware access | 🔴 Spec only |

## Future Install (not working yet)

```bash
pip install autogpt-local-executor
autogpt-shim auth    # OAuth flow → AutoGPT platform
autogpt-shim start   # Connect and run
```

## Docs

| Doc | Description |
|-----|-------------|
| [VISION.md](docs/VISION.md) | Full dream + platform changes needed per capability |
| [PROTOCOL.md](docs/PROTOCOL.md) | WebSocket message protocol spec |
| [OAUTH_FLOW.md](docs/OAUTH_FLOW.md) | Auth design (uses AutoGPT's existing OAuth provider) |
| [SECURITY.md](docs/SECURITY.md) | Threat model and defense layers |
| [PLATFORM_HOOKS.md](docs/PLATFORM_HOOKS.md) | Platform-side insertion points |
| [CROSS_PLATFORM.md](docs/CROSS_PLATFORM.md) | Per-OS support matrix and path-jail strategy |

## Contributing

Issues with `[local-executor]` prefix. PRs welcome — understand this interface will break repeatedly.
