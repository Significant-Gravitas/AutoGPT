# Platform Hooks — Where Local PC Executor Plugs Into AutoGPT

> **Status**: Spec / Not Implemented

This document maps every platform-side change needed to support the local PC executor.
All file paths are relative to `autogpt_platform/backend/backend/`.

---

## 1. `copilot/config.py` — New Config Fields

```python
# Add to CopilotConfig (or ChatConfig, whichever owns executor selection):

use_local_pc_executor: bool = Field(
    default=False,
    description="Route execution to a user's local machine via the LocalPC shim. "
                "EXPERIMENTAL. Overrides E2B when set.",
)

local_pc_allowed_root: str = Field(
    default="",
    description="Absolute path on the user's machine that the shim will jail file ops to. "
                "Advertised by shim in HELLO; platform validates against this.",
)

allow_computer_use: bool = Field(
    default=False,
    description="Allow Claude to take screenshots and inject mouse/keyboard input via the shim. "
                "Requires use_local_pc_executor=True. Off by default — explicit opt-in only.",
)

local_llm_policy: Literal["never", "prefer", "always"] = Field(
    default="never",
    description="Whether to route LLM inference to a local model on the shim.",
)

local_llm_model: str = Field(
    default="llama3.2:3b",
    description="Ollama model name to use when local_llm_policy != 'never'.",
)
```

---

## 2. `copilot/sdk/service.py` — `_setup_e2b()` → `_setup_executor()`

This is the **primary insertion point**. Currently at ~line 3815:

```python
# CURRENT:
async def _setup_e2b():
    if not (e2b_api_key := config.active_e2b_api_key):
        if config.use_e2b_sandbox:
            logger.warning("[E2B] no API key, falling back to bubblewrap")
        return None
    sandbox = await get_or_create_sandbox(session_id, api_key=e2b_api_key, ...)
    return sandbox

# PROPOSED — rename to _setup_executor(), add third branch:
async def _setup_executor():
    # Branch 1: Local PC shim (highest priority when configured)
    if config.use_local_pc_executor:
        shim = await get_or_create_local_pc_shim(session_id, user_id=user_id)
        if shim is not None:
            logger.info("[LocalPC] connected to shim %s", shim.machine_id)
            return shim
        logger.warning("[LocalPC] shim not connected, falling back to E2B/bubblewrap")

    # Branch 2: E2B cloud sandbox (existing)
    if not (e2b_api_key := config.active_e2b_api_key):
        if config.use_e2b_sandbox:
            logger.warning("[E2B] no API key, falling back to bubblewrap")
        return None
    sandbox = await get_or_create_sandbox(session_id, api_key=e2b_api_key, ...)
    return sandbox
```

`LocalPCShim` must satisfy the `AsyncSandbox` duck-type interface so the rest of the
function requires zero changes:

```python
# These lines work unchanged because LocalPCShim has .sandbox_id, .pause(), .kill():
use_executor = executor is not None
set_execution_context(user_id, session, sandbox=executor, ...)
mcp_server = create_copilot_mcp_server(use_e2b=use_executor)
```

Note: `create_copilot_mcp_server(use_e2b=...)` and `get_sdk_supplement(use_e2b=...)` may
need a rename or an additional `use_local_pc=...` param if the local PC path needs a
distinct MCP server configuration (e.g., to expose hardware tools).

---

## 3. New File: `copilot/tools/local_pc_shim.py`

The `LocalPCShim` class. Duck-typed to match E2B's `AsyncSandbox`.

See `shim/local_pc_shim.py` in this scaffold for the full skeleton.

Key interface:

```python
class LocalPCShim:
    sandbox_id: str              # = f"localpc:{machine_id}:{session_id}"
    machine_id: str
    capabilities: list[str]

    async def pause(self) -> None: ...
    async def kill(self) -> None: ...

    class commands:
        async def run(self, cmd: str, cwd: str | None, timeout: int) -> CommandResult: ...

    class files:
        async def read(self, path: str) -> bytes: ...
        async def write(self, path: str, content: bytes) -> None: ...
```

---

## 4. New File: `copilot/tools/local_pc_connection.py`

Manages the WebSocket connection pool:

```python
async def get_or_create_local_pc_shim(
    session_id: str,
    user_id: str,
) -> LocalPCShim | None:
    """
    Returns the active LocalPCShim for this session, or None if the user's
    shim is not currently connected.

    Redis key: copilot:localpc:shim:{session_id}
    Connection registry: in-memory dict on this worker (or Redis pub/sub for multi-worker)
    """
    ...
```

---

## 5. New WebSocket Route in `api/`

```python
# New file: api/routes/local_executor_ws.py

@router.websocket("/ws/local-executor/{session_id}")
async def local_executor_ws(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(...),   # Bearer token for the shim
):
    """
    Shim connects here. Platform:
    1. Validates token via existing introspect_token()
    2. Receives HELLO, validates capabilities
    3. Registers shim in ShimConnectionManager
    4. Relays EXECUTE_COMMAND / FILE_READ / FILE_WRITE messages
    5. On disconnect, marks session as shim-unavailable
    """
    ...
```

---

## 6. `copilot/context.py` — Allowed Dirs

```python
# CURRENT:
E2B_WORKDIR = "/home/user"
E2B_ALLOWED_DIRS = ("/home/user", "/tmp")

# ADD:
def get_allowed_dirs(sandbox) -> tuple[str, ...]:
    """Returns allowed dirs for the current executor."""
    from backend.copilot.tools.local_pc_shim import LocalPCShim
    if isinstance(sandbox, LocalPCShim):
        return (sandbox.allowed_root,)
    return E2B_ALLOWED_DIRS
```

---

## 7. `copilot/tools/e2b_file_tools.py` — Computer Use Integration

When `config.allow_computer_use` and the active sandbox is a `LocalPCShim`:

```python
# New handler registered alongside existing file tools:
async def _handle_screenshot(params, ...) -> ToolResult:
    sandbox = get_current_sandbox()
    if not isinstance(sandbox, LocalPCShim) or "computer_use" not in sandbox.capabilities:
        return error("Computer use not available in this session")
    screenshot_bytes = await sandbox.computer_use.screenshot()
    return image_result(screenshot_bytes, mime="image/jpeg")

async def _handle_input_action(params, ...) -> ToolResult:
    sandbox = get_current_sandbox()
    await sandbox.computer_use.execute_action(
        action=params["action"],
        coordinate=params.get("coordinate"),
        text=params.get("text"),
        key=params.get("key"),
    )
    return ok_result()
```

Claude's computer use tool calls flow through these handlers. The platform must pass
`betas=["computer-use-2025-11-24"]` and `tools=[{"type": "computer_20251124", ...}]`
to the Anthropic API when `allow_computer_use=True` — this happens in
`stream_chat_completion_sdk()` where the messages array is constructed.

---

## 8. Redis Key Namespace

```
copilot:e2b:sandbox:{session_id}          ← existing
copilot:localpc:shim:{session_id}         ← new: stores machine_id + connected_at
copilot:localpc:capabilities:{session_id} ← new: granted capabilities list
copilot:localpc:tasks:{user_id}           ← future: scheduled background tasks
```

---

## 9. Postgres Schema (Future — Background Tasks)

```sql
CREATE TABLE local_executor_shims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    machine_id TEXT NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    capabilities JSONB NOT NULL DEFAULT '[]',
    platform TEXT,
    shim_version TEXT,
    UNIQUE(user_id, machine_id)
);

CREATE TABLE local_executor_scheduled_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    shim_id UUID REFERENCES local_executor_shims(id),
    cron_expr TEXT,
    prompt TEXT NOT NULL,
    last_run_at TIMESTAMPTZ,
    next_run_at TIMESTAMPTZ,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT now()
);
```

---

## Change Summary

| File | Change Type | Priority |
|------|-------------|----------|
| `copilot/config.py` | Add fields | MVP |
| `copilot/sdk/service.py` | Add third executor branch | MVP |
| `copilot/tools/local_pc_shim.py` | New file | MVP |
| `copilot/tools/local_pc_connection.py` | New file | MVP |
| `api/routes/local_executor_ws.py` | New file | MVP |
| `copilot/context.py` | Dynamic allowed dirs | MVP |
| `copilot/tools/e2b_file_tools.py` | Computer use handlers | Computer Use milestone |
| `copilot/tools/e2b_file_tools.py` | Cross-OS adapter glue (see below) | MVP |
| Redis key namespace | New keys | MVP |
| Postgres schema | New tables | Background Tasks milestone |

---

## 10. Cross-OS Adapter Notes

The platform code today is implicitly POSIX-flavored. To make `LocalPCShim`
work on macOS/Linux/Windows/WSL2 the platform side needs the following
adjustments. Each item is a discrete work-list entry — none of them require
rewriting the executor abstraction.

### 10.1 Drop hard-coded `E2B_WORKDIR`

**Current**: `copilot/context.py` declares `E2B_WORKDIR = "/home/user"` and
several call sites assume that's the sandbox's working directory.

**Change**: when the active executor is a `LocalPCShim`, read the workdir
from `shim.allowed_root` (advertised in HELLO). The platform must NOT bake
in any POSIX path. The `get_allowed_dirs(sandbox)` helper from section 6
is the entry point — extend it to also return the workdir, or add
`get_workdir(sandbox)` next to it.

Callsites to audit: anywhere `E2B_WORKDIR` appears today, anywhere a tool
description string mentions `/home/user`, and anywhere a default `cwd` is
passed to `sandbox.commands.run(...)`.

### 10.2 Replace `readlink -f` with portable canonicalize

**Current**: at least one site (file resolution for symlink targets) calls
`sandbox.commands.run("readlink -f <path>")`. `readlink -f` does not exist
on macOS (BSD `readlink` has no `-f`) and does not exist on Windows.

**Change**: instead of running a shell command, call `FILE_STAT` with
`follow_symlinks: true` and read the resolved `path` field from the
response. The shim handles per-OS canonicalize correctly. Same applies to
any other shelled-out POSIX command name: `stat`, `ls`, `find`, `rm`, `mv`,
`mkdir -p`, `test -e` — all become FILE_STAT/FILE_LIST/FILE_DELETE/
FILE_MOVE protocol calls.

### 10.3 Allow shell selector pass-through

**Current**: at least one site builds the command with
`bash -c {shlex.quote(...)}`. This will fail on Windows (no bash by default)
and silently use POSIX `/bin/sh` on some non-bash macOS configurations.

**Change**: in the `LocalPCShim.commands.run` adapter, accept an optional
`shell=` kwarg (default `"auto"`) and forward it to the wire as the `shell`
field of EXECUTE_COMMAND. Even better: where the platform doesn't need a
shell (no pipes, no globs, no env-var expansion), pass `argv=[...]` to the
adapter and skip shell selection entirely. The adapter forwards `argv` as
the wire `argv` field.

Callsites to audit: anywhere the platform constructs a command string with
`bash -c`, `sh -c`, or `&&`/`||`/`|` operators. Convert to `argv` form
when possible.

### 10.4 Accept E2B-style kwargs on `LocalPCShim`

The `LocalPCShim` class duck-types `e2b.AsyncSandbox`. E2B's kwargs differ
from the shim's wire field names; the adapter must translate transparently
so platform code is identical whether the executor is E2B or LocalPC.

| Adapter method | E2B kwarg | Wire field | Translation in adapter |
|---|---|---|---|
| `commands.run(...)` | `envs=` | `env` | `payload["env"] = kwargs.pop("envs", None) or kwargs.get("env")` |
| `commands.run(...)` | `timeout=` (seconds) | `timeout_seconds` | rename in adapter |
| `commands.run(...)` | `cwd=` | `cwd` | passthrough |
| `files.read(path, format=...)` | `format="text"` / `"bytes"` | `encoding` + `format` | see PROTOCOL.md mapping table |
| `files.write(path, data, ...)` | `data` (bytes or str) | `content` + `encoding` | base64-encode bytes; pass strings as utf-8 |

**Do not** change platform code to use the wire names. The whole point of
duck-typing E2B is that platform code remains unchanged. The translation
happens inside `LocalPCShim.commands` and `LocalPCShim.files`.

### 10.5 Advertise platform / arch to the UI

The HELLO message contains `platform` and `arch`. Surface these in any UI
or log line that mentions "connected to local PC" — users will need to know
"shim running on Windows" so they can interpret error messages. Suggested
display strings:

| Wire value | Display |
|---|---|
| `darwin` | "macOS" |
| `linux` | "Linux" |
| `windows` | "Windows" |
| `wsl2` | "Windows (WSL2)" |

### 10.6 Don't assume case-sensitive paths

Platform code that compares paths (e.g., "is this file inside the workdir")
must NOT use `==` or `.startswith` directly. On macOS and Windows the same
file has multiple valid string forms (`/Users/Alice/Foo` vs
`/users/alice/foo`). Either:

1. Send the comparison to the shim via `FILE_STAT` and use the resolved
   path the shim returns, OR
2. If comparing in platform code, lowercase both sides BUT acknowledge this
   is best-effort and is no substitute for a shim-side check.

### 10.7 Strip CRLF assumptions

Platform code that processes file content read from the shim must tolerate
both `\n` and `\r\n` line endings. The shim returns bytes as-is from disk
(see PROTOCOL.md). If the platform needs canonical `\n`-only content, it
must normalize at the platform side, not assume the shim has done it.

### 10.8 OAuth callback port robustness

The fixed port `41899` may be in use by another app. The OAuth flow doc
(OAUTH_FLOW.md) specifies a scan range of `41899–41910`. The platform's
OAuth client registration must accept ALL ports in that range as valid
`redirect_uri`s for `client_id=autogpt-local-executor`, not just `41899`.
