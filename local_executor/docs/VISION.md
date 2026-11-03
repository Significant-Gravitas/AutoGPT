# Vision: AutoGPT Local PC Executor
## The Dream — Where This Could Go

> **Status**: Experimental / Pre-Alpha Spec  
> **Warning**: This document describes aspirational goals. Nothing here is built yet.

---

## What We Have Today

AutoGPT runs all code execution inside **E2B cloud sandboxes** — ephemeral Linux containers
that Claude can read/write files in, run shell commands, and use as a workspace. When E2B
isn't available, we fall back to **bubblewrap** (`bwrap`), a local Linux sandbox.

The problem: neither of these can touch *your actual machine*. Your files, your apps,
your hardware — all behind a wall.

---

## The Core Idea

A lightweight **local shim daemon** running on the user's machine. It opens an outbound
WebSocket to the AutoGPT platform and presents as a `LocalPCShim` object that is
**duck-type compatible** with the E2B `AsyncSandbox` interface. From the platform's
perspective, it just looks like another sandbox.

```
[AutoGPT Cloud Platform]
        |
        |  (existing WebSocket infra)
        |
[LocalPCShim daemon — user's machine]
        |
    user's actual filesystem, processes, hardware
```

No inbound ports required. No firewall rules. The user installs the shim, authenticates
via the existing AutoGPT OAuth flow, and their machine becomes an executor.

---

## Platform Changes Required for Each Dream

### 1. Basic Shell + File Access (MVP)
**Dream**: Run commands on your real machine, read/write real files.

**Platform changes needed**:
- `config.py`: Add `use_local_pc_executor: bool` and `local_pc_allowed_root: str` fields
- `service.py` / `_setup_e2b()`: Add third branch — if `config.use_local_pc_executor`, construct `LocalPCShim` and return it instead of E2B sandbox
- `context.py`: `E2B_ALLOWED_DIRS` becomes per-executor config; shim advertises its own `allowed_root`
- `e2b_file_tools.py`: Already branches on `get_current_sandbox()` — `LocalPCShim` satisfying the duck-type interface means zero changes needed here
- `config.py`: New `local_pc_executor_token_secret` for WebSocket HMAC validation

**Shim changes needed**:
- `ShimCommandRunner`: wraps `subprocess` (or optionally `bwrap` for sandboxing within the shim)
- `ShimFileStore`: wraps `pathlib` with path-jail enforcement against `allowed_root`
- WebSocket message loop: deserializes `ExecuteCommand | ReadFile | WriteFile` messages, runs them, sends back `CommandResult | FileContents | Ack`

---

### 2. Computer Use — Screen Vision + Input Injection
**Dream**: Claude sees your screen, moves your mouse, types on your keyboard. Full GUI automation on real apps.

**How Anthropic's computer use API works**:
Claude's `computer_20251124` tool (beta) operates a 4-step loop:
1. Claude requests a `screenshot` → shim captures screen and returns base64 PNG
2. Claude analyzes pixel coordinates of UI elements
3. Claude issues `left_click([x, y])`, `type("text")`, `key("ctrl+s")`, `mouse_move([x,y])`, `scroll([x,y], direction)`
4. Repeat until task done

The shim executes these using `pyautogui` (cross-platform) or `xdotool` (Linux X11).

**Platform changes needed**:
- `PROTOCOL.md`: Extend `MessageType` enum with `SCREENSHOT_REQUEST`, `SCREENSHOT_RESPONSE`, `INPUT_ACTION`, `INPUT_ACK`
- `service.py`: When `config.use_local_pc_executor and config.allow_computer_use`, pass `betas=["computer-use-2025-11-24"]` and `tools=[{"type": "computer_20251124"}]` to the Claude API call (currently in `stream_chat_completion_sdk`)
- New `ComputerUseRouter` class in platform: intercepts Claude's computer use tool calls, forwards screenshot requests to shim WebSocket, returns image back to Claude as `image/base64`
- `ChatConfig`: Add `allow_computer_use: bool = False` gate — off by default, user must explicitly enable
- Session capability advertisement: shim's `HELLO` handshake message must include `"capabilities": ["shell", "files", "computer_use"]`

**Shim changes needed**:
- `ShimComputerUse`: wraps `pyautogui.screenshot()` → JPEG compress → base64 → send
- Action executor: `pyautogui.click()`, `pyautogui.write()`, `pyautogui.hotkey()`, `pyautogui.scroll()`
- Screen resolution advertisement in `HELLO` message so Claude can reason about coordinates
- Optional: multi-monitor support, specific window targeting

**Security gate**: Computer use must be an explicit opt-in with a confirmation dialog in the platform UI. The user should see "Claude is requesting screen access" and approve each session.

---

### 3. Resident Agent Daemon — Always-On Background Tasks
**Dream**: The shim stays running as a system service. Claude can schedule tasks, watch for file changes, react to system events — without the user being present.

**Platform changes needed**:
- `executor/processor.py`: Add concept of "shim-routed turns" — turns that don't need the user to be in the chat loop
- New `ShimTaskScheduler` service: persists scheduled tasks to `copilot:localpc:tasks:{user_id}` in Redis
- `CoPilotProcessor`: Add `background_mode: bool` flag — suppresses streaming output, stores result summary instead
- Webhook endpoint: `POST /copilot/local-executor/event` — shim can push filesystem events, which enqueue a new copilot turn
- New Postgres table: `local_executor_tasks(id, user_id, shim_id, cron_expr, prompt, last_run, next_run, status)`

**Shim changes needed**:
- `ShimWatcher`: `watchdog`-based filesystem watcher, pushes `FILE_CHANGED` events over WebSocket
- `ShimScheduler`: cron runner that wakes up and sends `TRIGGER_TURN` messages on schedule
- `systemd` / `launchd` / Windows Service installer for the shim daemon
- Keepalive: exponential backoff reconnect loop with jitter so all shims don't reconnect simultaneously after a platform restart

---

### 4. Hardware Access — USB, GPIO, Serial, Printers
**Dream**: Claude controls physical hardware — 3D printers, Arduino/Raspberry Pi GPIO, USB devices, serial ports, lab instruments (GPIB/VISA).

**Platform changes needed**:
- `PROTOCOL.md`: New `MessageType.HARDWARE_OP` with subtypes: `SERIAL_WRITE`, `SERIAL_READ`, `USB_SEND`, `GPIO_SET`, `GPIO_GET`
- Capability advertisement: shim's `HELLO` lists `"hardware": ["serial:/dev/ttyUSB0", "gpio:bcm", "usb:2341:0043"]`
- Platform UI: hardware capability browser — user sees what devices are available and grants per-device access to Claude
- `ChatConfig`: `allowed_hardware_devices: list[str]` scoped per session

**Shim changes needed**:
- `ShimSerialPort`: wraps `pyserial` — `serial.Serial(port, baud).write/read`
- `ShimGPIO`: wraps `RPi.GPIO` or `gpiozero` for Raspberry Pi
- `ShimUSB`: wraps `pyusb` for raw USB HID/bulk transfers
- `ShimPrinter`: wraps OctoPrint REST API for 3D printers
- Device enumeration at startup: `usb.core.find(find_all=True)`, `serial.tools.list_ports.comports()`

---

### 5. Privacy Inversion — Files Never Leave the Machine
**Dream**: Claude reasons about sensitive data (medical records, financial files, private code) without that data ever hitting the cloud. The LLM prompt references file *metadata and structure*, not raw content.

**Platform changes needed**:
- `e2b_file_tools.py` / `_handle_read_file`: New "privacy mode" path — instead of returning file content, return a `FileStub` with `{path, size, mime_type, hash, schema_preview}` and send the full content *only* to a local LLM running on the shim
- New `PrivacyRouter` in `service.py`: for privacy-flagged sessions, file tool results are stubs; a second local-LLM turn runs on the shim to process the actual content and returns only the *answer*, not the content
- `ChatConfig`: `privacy_mode: bool = False` — enables content-blind routing
- New message type `LOCAL_LLM_TURN`: platform sends a prompt fragment + file stub; shim resolves the file locally, calls local LLM, returns the response

**Shim changes needed**:
- `ShimLocalLLM`: wraps Ollama HTTP API (`POST http://localhost:11434/api/chat`)
- File content never serialized into WebSocket messages in privacy mode
- Local inference result summary sent back instead

---

### 6. Multi-Machine Orchestration
**Dream**: One AutoGPT session coordinates work across multiple machines — dev box, build server, NAS, Raspberry Pi cluster.

**Platform changes needed**:
- `session_id → [shim_id]` mapping in Redis: one session can have multiple shims
- `LocalPCRouter`: given a tool call, routes to the right shim based on `target_machine` hint in the tool call or a routing policy
- Claude tool augmentation: expose a `list_machines()` tool that returns shim capability advertisements
- `PROTOCOL.md`: Shims must register with a `machine_id` (hostname + UUID); platform maps `session → {machine_id: shim_ws_connection}`

**Shim changes needed**:
- No changes to individual shims — they just need stable `machine_id` in `HELLO`
- Optional: shim-to-shim direct tunnel for large file transfers (avoids routing through platform)

---

### 7. Local LLM Routing (Ollama / llama.cpp / LM Studio)
**Dream**: For tasks that don't need Claude's full capability, route inference to a local model. Lower latency, no API costs, works offline.

**Platform changes needed**:
- `stream_chat_completion_sdk()` in `service.py`: New routing layer — if `config.prefer_local_llm` and shim advertises `"local_llm": true`, send the turn to shim instead of Anthropic API
- `LocalLLMAdapter`: wraps shim's local LLM response into the same `StreamedEvent` interface the rest of the platform expects
- Model capability tiers: Claude routes to local for sub-tasks (summarize, classify, extract), keeps Anthropic API for reasoning-heavy turns
- `ChatConfig`: `local_llm_policy: "never" | "prefer" | "always"` + `local_llm_model: str`

**Shim changes needed**:
- `ShimLocalLLM.stream_chat()`: calls `ollama.chat(model=..., messages=..., stream=True)` and streams tokens back over WebSocket
- Model management: `ollama pull`, `ollama list` wrapped as platform tool calls

---

## The Execution Architecture We're Targeting

```
Platform (cloud)
├── CoPilotProcessor
│   └── stream_chat_completion_sdk()
│       ├── _setup_executor()          ← renamed from _setup_e2b()
│       │   ├── E2B sandbox            (existing)
│       │   ├── bubblewrap             (existing fallback)
│       │   └── LocalPCShim           ← NEW
│       │       ├── .commands.run()   ← WebSocket → shim ShimCommandRunner
│       │       ├── .files.read()     ← WebSocket → shim ShimFileStore
│       │       ├── .files.write()    ← WebSocket → shim ShimFileStore
│       │       ├── .computer_use()   ← WebSocket → shim ShimComputerUse (optional)
│       │       ├── .hardware_op()    ← WebSocket → shim ShimHardware (optional)
│       │       └── .local_llm()      ← WebSocket → shim ShimLocalLLM (optional)
│       └── set_execution_context(sandbox=shim)
│           └── e2b_file_tools.py     ← zero changes, uses duck-typed interface
│
└── WebSocket endpoint: /ws/local-executor/{session_id}
    └── ShimConnectionManager
        ├── Authenticates via introspect_token()
        ├── Registers shim capabilities
        └── Multiplexes tool calls ↔ results
```

---

## What Makes This Powerful vs. Just Running Code Locally

The key insight is that **the LLM remains in the cloud** but **execution happens locally**. This means:

1. **Full Claude capability** — no downgrade to a smaller local model (unless you want that)
2. **Platform memory + agent graphs** — all AutoGPT orchestration features work
3. **Shareable sessions** — collaborate with teammates on tasks that run on your machine
4. **Audit trail** — every command the shim executes is logged to the platform
5. **Granular permissions** — you can allow file access but block network, or allow reading but not writing
6. **Zero new infrastructure** — uses existing OAuth, existing WebSocket patterns, existing executor interface

The shim is intentionally **thin**. It doesn't make decisions. It just executes what the platform
tells it to, within the bounds of what the user has permitted.

---

## Open Questions (Pre-Alpha)

- [ ] Should computer use require a separate user confirmation per-session or per-action?
- [ ] How do we handle shim disconnects mid-turn? (Partial command execution, file half-written)
- [ ] Should the shim support Windows/macOS in v1 or Linux-only first?
- [ ] Multi-tenant: can a user share their shim with other users in their org?
- [ ] Audit log granularity: log every command? Every file read? Or just turn-level summaries?
- [ ] Rate limiting: should the platform throttle how fast it can issue commands to avoid runaway agents?

---

*This document lives in `experimental/local-pc-executor/docs/VISION.md`. Update it as the spec evolves.*
