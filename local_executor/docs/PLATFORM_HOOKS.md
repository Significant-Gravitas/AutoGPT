# Platform Hooks — Where Local PC Executor Plugs Into AutoGPT

> **Status**: the platform side is built. It merged into the AutoGPT
> monorepo's `dev` branch as
> [#13050](https://github.com/Significant-Gravitas/AutoGPT/pull/13050).
>
> Sections 1 to 9 are the design sketch that was written before it, kept for
> its reasoning. Each starts with an **As built** note that says what #13050
> does; where a sketch and its note differ, the note is right, and the code
> in a sketch is not the platform's. Section 10 was the platform's work list
> for other operating systems: each entry says how it landed.

This document maps the platform-side changes that support the local PC executor.
All file paths are relative to `autogpt_platform/backend/backend/`.

---

## 1. `copilot/config.py` — New Config Fields

> **As built (#13050):** the rollout gates are per-user PostHog flags, not
> config fields: `local-pc-executor` for Local PC and
> `local-pc-computer-use` for computer use (§10.9), read through
> `api/features/local_executor/gating.py`. Whether a chat runs on a computer
> is that chat's own setting, its `execution_target` (the computer's
> `machine_id` and the shared folder), which the person picks when they
> start the chat. `ChatConfig` gained two fields:
> `local_pc_executor_oauth_client_id` (the one OAuth client whose tokens
> may connect a computer, `autogpt-local-executor` by default) and
> `local_llm_policy` (`never`, `prefer_for_fast` or `always`; `never` by
> default). Of the fields sketched below, that last one alone was added,
> with `prefer_for_fast` where the sketch has `prefer`; none of the other
> four was.

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

> **As built (#13050):** `_setup_e2b()` kept its name. For a chat whose
> `execution_target.kind` is `local` it calls `_setup_local_pc_executor()`,
> which returns the `LocalPCShim` for the computer and folder the chat is
> bound to (`LocalPCShim.for_session`). It never falls back to the cloud: a
> Local PC chat whose computer isn't connected, or doesn't match the chat's
> binding, gets no executor, and the turn ends with
> `local_pc_executor_unavailable`. The fallback in the sketch below would
> have run a Local PC chat's commands in a cloud sandbox instead.

This was the **primary insertion point** in the sketch:

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
distinct MCP server configuration (e.g., to expose hardware tools). As built,
`create_copilot_mcp_server` takes `use_local_pc_computer=` for the computer-use
tools (§10.9); there are no hardware tools.

---

## 3. New File: `copilot/tools/local_pc_shim.py`

> **As built (#13050):** the file exists under this name, and `LocalPCShim`
> has the shape below: `commands.run`, `files.read` and `files.write`,
> `pause()` and `kill()`. Its `sandbox_id` is the chat's session ID, not
> the `localpc:` form in the sketch. It also carries the computer-use and
> recording calls, and the errors a caller tells apart
> (`copilot/tools/local_pc_errors.py`).

The `LocalPCShim` class. Duck-typed to match E2B's `AsyncSandbox`.

Key interface, as sketched:

```python
class LocalPCShim:
    sandbox_id: str  # = f"localpc:{machine_id}:{session_id}"
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

> **As built (#13050):** there is no `local_pc_connection.py`. The worker
> that holds a computer's socket keeps it in `ShimConnectionManager`
> (`copilot/tools/local_pc_shim.py`), and the `copilot/tools/local_pc_relay*.py`
> modules carry requests and replies between that worker and the one
> running a chat's turn, over Redis (§8). A turn gets its `LocalPCShim`
> from `LocalPCShim.for_session(session_id, ...)`.

The sketch managed the WebSocket connection pool like this:

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

> **As built (#13050):** `api/features/local_executor/websocket.py`, with
> the HTTP routes beside it in `routes.py`. This section shows the routes as
> merged. The sketch that stood here took the token as a query parameter,
> which the platform does not accept: a token in a URL is kept by proxies
> and access logs, and this one is long-lived.

```python
# api/features/local_executor/websocket.py


@router.websocket("/ws/local-executor")
async def local_executor_machine_ws(websocket: WebSocket) -> None:
    """One connection per computer, kept across chats."""


@router.websocket("/ws/local-executor/{session_id}")
async def local_executor_ws(session_id: SessionID, websocket: WebSocket) -> None:
    """One connection per chat that runs on the computer."""


def _bearer_token(websocket: WebSocket) -> str | None:
    # The handshake's `Authorization: Bearer <access token>` header, and
    # nothing else: no query parameter, no cookie, no subprotocol.
    authorization = websocket.headers.get("authorization", "")
    scheme, separator, credentials = authorization.partition(" ")
    if separator and scheme.lower() == "bearer" and credentials.strip():
        return credentials.strip()
    return None
```

Both routes check the same things before they accept the connection
(`_authenticate_connection`):

1. The access token is in the `Authorization` header. A connection without
   that header is refused with "Missing bearer token" (HTTP 401, or close
   code 4401 where the server can't answer the handshake with a status),
   also when the URL carries `?token=...`
   (`routes_test.py::test_query_string_token_is_not_accepted`).
2. `introspect_token()` says it is an active, unexpired access token for a
   user, issued to the client named by `local_pc_executor_oauth_client_id`.
3. It has the `CONNECT_LOCAL_EXECUTOR` scope. A token without it is refused
   with 403, which the shim doesn't retry: refreshing keeps a token's scopes.
4. For a chat's connection, the chat is that user's and is bound to a
   Local PC.
5. The user's `local-pc-executor` flag is on.

After the HELLO it relays the protocol's messages, and it closes the
connection when the token expires (4401), when it is revoked (4428, which
tells the shim not to reconnect) or when the flag is turned off (4403).
[PROTOCOL.md](PROTOCOL.md) lists the close codes after which the shim
doesn't reconnect (4426 to 4428).

---

## 6. `copilot/context.py` — Allowed Dirs

> **As built (#13050):** `get_allowed_dirs(sandbox)` and `get_workdir(sandbox)`
> are in `copilot/context.py`. For a Local PC they return the folder the
> computer advertised in HELLO (`allowed_root`), with no `/tmp` beside it;
> for E2B they return the fixed `/home/user` and `/tmp`. A Local PC is told
> apart by `is_local_pc_executor(sandbox)`, so `context.py` doesn't import
> `LocalPCShim` at run time as the sketch does.

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

## 7. Computer Use Integration

Superseded by the implementation in [§10.9](#109-computer-use-routing-platform-side-in-13050):
the shim relays cua-driver's tools ([CUA_DRIVER.md](CUA_DRIVER.md)) and the
platform exposes them to Otto through two MCP tools, not the native
`computer_20251124` tool sketched here.

---

## 8. Redis Key Namespace

> **As built (#13050):** none of the sketched `copilot:localpc:*` keys exists.
> The keys the platform uses are under `local-executor:`, with the chat's or
> the computer's tag in braces so that a cluster keeps each one's keys on
> one node: a `...:presence` hash and a per-owner `...:presences` index for
> what is connected, one stream per connection and direction for the
> requests and replies relayed between workers
> (`copilot/tools/local_pc_relay_protocol.py`), and the recording state
> (`api/features/local_executor/state.py`). Consent to computer use is under
> `copilot:local-executor:computer-use-consent:`
> (`api/features/local_executor/consent.py`).

The sketch:

```
copilot:e2b:sandbox:{session_id}          ← existing
copilot:localpc:shim:{session_id}         ← new: stores machine_id + connected_at
copilot:localpc:capabilities:{session_id} ← new: granted capabilities list
copilot:localpc:tasks:{user_id}           ← future: scheduled background tasks
```

---

## 9. Postgres Schema (Future — Background Tasks)

> **As built (#13050):** not built. There is no table for computers or for
> scheduled tasks: what is connected lives in Redis (§8), and a chat's
> binding to a computer and folder is in the chat's own metadata. #13050's
> two migrations add `OAuthApplication.isPublic`
> (`20261102120000_add_oauth_application_is_public`) and the
> `CONNECT_LOCAL_EXECUTOR` permission
> (`20261103210000_add_connect_local_executor_permission`); see
> [OAUTH_FLOW.md](OAUTH_FLOW.md).

The sketch, for background tasks that don't exist yet:

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

| Sketched | As built (#13050) |
|------|-------------|
| `copilot/config.py`: new fields | PostHog flags, read by `api/features/local_executor/gating.py`; `ChatConfig.local_pc_executor_oauth_client_id` and `local_llm_policy` (§1) |
| `copilot/sdk/service.py`: a third executor branch | `_setup_e2b()` calls `_setup_local_pc_executor()` for a Local PC chat, with no fallback to the cloud (§2) |
| `copilot/tools/local_pc_shim.py`: new file | That file (§3) |
| `copilot/tools/local_pc_connection.py`: new file | `ShimConnectionManager` in `local_pc_shim.py`, and `copilot/tools/local_pc_relay*.py` (§4) |
| `api/routes/local_executor_ws.py`: new file | `api/features/local_executor/websocket.py`, with `routes.py` for the HTTP API (§5) |
| `copilot/context.py`: dynamic allowed dirs | `get_allowed_dirs()` and `get_workdir()` (§6) |
| `copilot/tools/e2b_file_tools.py`: computer use handlers | `copilot/sdk/computer_use_tools.py` (§10.9) |
| `copilot/tools/e2b_file_tools.py`: cross-OS adapter glue | `copilot/sdk/local_pc_file_tools.py` and `LocalPCShim` (§10) |
| Redis key namespace | Other keys (§8) |
| Postgres schema | Not built (§9) |

---

## 10. Cross-OS Adapter Notes

This was the platform's work list for making `LocalPCShim` work on
macOS/Linux/Windows/WSL2, written while the platform code was implicitly
POSIX-flavored. Each entry keeps its reasoning ("Current" is the platform
before #13050) and ends with how #13050 did it.

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

**As built (#13050):** `get_workdir(sandbox)` in `copilot/context.py` returns
the computer's `allowed_root` for a Local PC and `E2B_WORKDIR` otherwise.

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

**As built (#13050):** a Local PC's file tools are
`copilot/sdk/local_pc_file_tools.py`, which make those protocol calls; the
E2B sandbox keeps `readlink -f`.

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

**As built (#13050):** `LocalPCShim.commands.run(command, *, argv=None,
shell="auto", cwd=None, timeout=None, envs=None)`. With `argv` it sends
`argv` and no shell; otherwise it sends `command` with `shell`.

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

**As built (#13050):** `commands.run` takes `envs=` and `timeout=` and sends
`env` and `timeout_seconds`; `files.read(path, format="text")` and
`files.write(path, content)` take E2B's arguments.

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

**As built (#13050):** the frontend's `LocalPCBadge` and the folder dialog
show these names.

### 10.6 Don't assume case-sensitive paths

Platform code that compares paths (e.g., "is this file inside the workdir")
must NOT use `==` or `.startswith` directly. On macOS and Windows the same
file has multiple valid string forms (`/Users/Alice/Foo` vs
`/users/alice/foo`). Either:

1. Send the comparison to the shim via `FILE_STAT` and use the resolved
   path the shim returns, OR
2. If comparing in platform code, lowercase both sides BUT acknowledge this
   is best-effort and is no substitute for a shim-side check.

**As built (#13050):** `resolve_executor_path()` in `copilot/context.py`
checks a path against the shared folder by its text only, with Windows'
path rules for a Windows computer, to catch an obviously outside path
before a round trip. The shim's own jail is the boundary; what it does is in
[CROSS_PLATFORM.md](CROSS_PLATFORM.md#path-handling).

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

**As built (#13050):** a redirect URI must equal one the client registered,
so the registration lists all twelve. The command in
[OAUTH_FLOW.md](OAUTH_FLOW.md) does, and so does the registration that
preview deployments seed (`api/features/local_executor/preview_seed.py`).

### 10.9 Computer-use routing (platform side in #13050)

The platform side merged with #13050, and the platform files named below
are on `dev`. When all of — the
per-user `local-pc-executor` and `local-pc-computer-use` PostHog flags
AND the user's consent for the machine AND the connected shim advertised
`"computer_use"` in its HELLO `capabilities` — are true, Otto's computer
use runs on the user's shim through cua-driver.

How the four sketched pieces land there:

1. **CLI beta flag pass-through.**
   `backend/copilot/sdk/env.py::build_sdk_env(enable_computer_use_beta=…)`
   conditionally appends `computer-use-2025-11-24` to `ANTHROPIC_BETAS`
   (composable with any pre-existing betas) and drops
   `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` — **unless** the active
   transport is OpenRouter, in which case the strip flag stays
   unconditionally (OpenRouter cannot proxy computer-use; honest
   degradation rather than silent 4xx). `env_test.py` covers each
   branch.

2. **Tool-call surface: cua-driver's tools behind two MCP tools.** The
   shim relays a reviewed set of cua-driver's tools (protocol 2.0,
   [CUA_DRIVER.md](CUA_DRIVER.md)). Registering each as an MCP tool would
   put ~40 large schemas into every turn, so
   `backend/copilot/sdk/computer_use_tools.py` registers two:
   `local_pc_computer_tools` returns the driver's operating guidance, the
   tools this computer offers and the input schemas Otto asks for (cached
   per connection by `_ComputerProxy.tools()`), and `local_pc_computer`
   relays one call (`_ComputerProxy.call()`, never retried). Results carry
   the driver's text and screenshots as MCP content, then its structured
   content without strings the text already shows; a driver error is an
   `isError` result Otto reads, a shim ERROR a structured `{code, error,
   details}` payload translated by `local_pc_errors`. Only
   `local_pc_computer` carries the mutating annotation, so the CLI
   serialises actions on one screen.

   Registration is gated by `create_copilot_mcp_server(use_local_pc_computer=…)`
   in `tool_adapter.py`. The CLI's `allowed_tools` whitelist
   (`get_copilot_tool_names`) and the permissions-gated path in
   `service.py` both append the two names under the same gate, so a
   misconfigured caller fails closed.

3. **Consent.** `HELLO.computer_use_driver` and `HELLO.computer_use_tools`
   define what approving grants. Consent is server-side and per chat
   session (`api/features/local_executor/consent.py`), bound to the
   machine, the driver release and the exact tool names; a shim that
   reconnects offering more asks again. Every call re-checks it, and the
   connection guard catches a reconnect in between. The frontend's
   `LocalPCComputerUseConsent` dialog describes the offered tools in plain
   groups and posts the scope it showed (`expected_machine_id`,
   `expected_driver`, `expected_tools`); the backend refuses a scope that
   changed. **The backend gate is the security boundary; the dialog makes
   it visible.**

   A call that uses one of a tool's `approval_arguments` (disk paths, a
   `run_script` script, or what `launch_app` would run) goes through the
   chat's approval gate on its own subject, and only once the gate approves
   it does the platform send it with `approved_arguments`. Usually the
   person approves it on a card that shows the path, script or program.
   Two cases approve it without a card for that call: a chat that runs
   Unsupervised approves these calls itself, and a standing rule the person
   set on an earlier card ("from now on") approves later calls of that kind
   whatever their arguments. Where the gate can't ask and neither applies,
   the platform refuses the call. The shim can't tell how a call was
   approved: it trusts `approved_arguments`, and refuses a call without it.

4. **Tests.** `env_test.py` covers the OpenRouter compatibility
   regression. `local_pc_shim_test.py::TestComputerProxy` covers the two
   wire messages, `computer_use_tools_test.py` the MCP handlers and their
   gate, `consent_test.py` and `routes_api_test.py` the consent scope, and
   the shim's `tests/test_cua_driver*.py` the relay against a fake and the
   real driver.
