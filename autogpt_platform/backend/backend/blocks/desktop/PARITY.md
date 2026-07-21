# Desktop Sandbox Parity Contract (E2B ⇄ Daytona)

Two branches implement the SAME feature surface with different sandbox providers, for a
head-to-head capability + cost comparison:

- `ntindle/sandbox-e2b-desktop` — keeps the existing E2B integration, ADDS interactive
  desktops + suspend/resume persistence + mounted workspace volumes.
- `ntindle/sandbox-daytona` — FULL migration: every sandbox-backed feature (code executor
  blocks, claude_code, sandbox files util, desktop blocks) backed by Daytona instead of E2B.

Same block IDs + same input/output schemas on both branches, so one exported agent graph
runs on either branch unchanged. If an implementation detail in this contract proves
unworkable on one provider, do NOT silently diverge — document the deviation in
`PARITY_NOTES.md` next to this file.

## Blocks (both branches, `backend/blocks/desktop/`)

Fixed IDs (valid uuid4 format — required by `test_block_ids_valid`):

| Block class | ID | Purpose |
|---|---|---|
| `CreateDesktopSandboxBlock` | `a1e2b001-0001-4000-8000-de5c704b0001` | Create or reconnect a desktop sandbox; mount persistent workspace volume; start interactive stream |
| `DesktopActionBlock` | `a1e2b001-0002-4000-8000-de5c704b0002` | Computer-use primitive: screenshot / clicks / move / drag / scroll / type / press / hotkey / wait |
| `DesktopCommandBlock` | `a1e2b001-0003-4000-8000-de5c704b0003` | Run a shell command inside the desktop sandbox |
| `DesktopFileBlock` | `a1e2b001-0004-4000-8000-de5c704b0004` | Read / write / list files in the sandbox |
| `StopDesktopSandboxBlock` | `a1e2b001-0005-4000-8000-de5c704b0005` | Suspend (default, preserves state) or destroy the sandbox |

Common rules:

- Credentials input: `CredentialsMetaInput[Literal[ProviderName.E2B], Literal["api_key"]]`
  on the E2B branch; `ProviderName.DAYTONA` on the Daytona branch. Everything else in the
  schemas MUST be identical.
- Every block needs `test_input` / `test_output` / `test_mock` (enforced by
  `test_available_blocks`) and regenerated docs
  (`poetry run python scripts/generate_block_docs.py`, enforced by docs-block-sync CI).
- Category: `BlockCategory.DEVELOPER_TOOLS` (+ `MULTIMEDIA` for the stream output block if
  it fits existing conventions).

### CreateDesktopSandboxBlock

Inputs:
- `credentials`
- `workspace_scope`: enum `user` | `agent` (default `user`). Selects the persistent volume
  identity: `autogpt-user-{user_id}` or `autogpt-agent-{graph_id}` (ids taken from
  execution context, not user input). Future agent variants (copilot/autopilot instances)
  each get a durable workspace; per-user is a durable "my computer".
- `sandbox_id` (optional, default ""): reconnect/resume an existing desktop instead of
  creating a new one.
- `width`: int, default 1280; `height`: int, default 720 (flat fields, not a nested
  object — wires better in the builder).
- `timeout_minutes`: int, default 15 — idle timeout after which the sandbox auto-suspends
  (E2B: `SandboxLifecycle(on_timeout="pause", auto_resume=True)`; Daytona:
  `auto_stop_interval`).

Outputs:
- `sandbox_id`: str
- `desktop_stream`: object — THE parity-critical shape the frontend renderer keys on:
  ```json
  {
    "kind": "desktop_stream",
    "url": "<directly iframe-embeddable URL>",
    "provider": "e2b" | "daytona",
    "sandbox_id": "...",
    "requires_auth": false
  }
  ```
  E2B: `stream.start(require_auth=True)` + `stream.get_url(auth_key=...)`.
  Daytona: `computer_use.start()` + signed preview URL for the noVNC port
  (`create_signed_preview_url(6080, ...)`) so it works in an iframe without headers.
- `workspace_path`: str — persistent dir inside the sandbox (`/home/user/workspace` E2B,
  `/home/daytona/workspace` Daytona)
- `persistence`: object `{volume_mounted: bool, volume_name: str|null, warning: str|null}`
  (see Persistence model)
- `cost_meter`: object (see Cost telemetry)
- `error`: str

### DesktopActionBlock

Inputs (FLAT fields — a nested discriminated union wires poorly in the builder):
`credentials`, `sandbox_id`, `action` enum:
`screenshot` | `left_click` | `double_click` | `right_click` | `middle_click` |
`move` | `drag` | `scroll` | `type` | `press` | `wait`;
plus `x`, `y` (Optional[int] — click/move/drag start), `to_x`, `to_y` (drag
destination), `text` (type), `keys: list[str]` (press — single key or combo, e.g.
`["ctrl","c"]`; there is no separate hotkey action), `scroll_direction`
(`up`|`down`, default down), `scroll_amount` (default 3), `seconds` (wait, capped
at 60), `screenshot_after: bool` (default true).
Outputs: `result` (str), `screenshot` (image media output via `store_media_file`
`for_block_output`), `cost_meter`, `error`.

### DesktopCommandBlock

Inputs: `credentials`, `sandbox_id`, `command`, `cwd` (default = workspace path),
`timeout_seconds` (default 60).
Outputs: `stdout`, `stderr`, `exit_code`, `cost_meter`, `error`.

### DesktopFileBlock

Inputs: `credentials`, `sandbox_id`, `operation` (`read`|`write`|`list`), `path`,
`content` (write only).
Outputs: `content` (read), `entries: list[str]` (list), `path`, `cost_meter`, `error`.

### StopDesktopSandboxBlock

Inputs: `credentials`, `sandbox_id`, `mode`: `suspend` (default) | `destroy`.
- E2B: suspend = `pause()` (state kept indefinitely, resume via `connect`); destroy = `kill()`.
- Daytona: suspend = `stop()` (filesystem persists, resume via `start()`); destroy = `delete()`.
Outputs: `sandbox_id`, `final_status` (`suspended`|`destroyed`), `cost_meter`, `error`.

## Persistence model (two layers, both branches)

1. **Sandbox suspend/resume** — whole-machine state via the provider's native mechanism
   (E2B pause/connect with `SandboxLifecycle` auto-pause — see
   `backend/copilot/tools/e2b_sandbox.py` for the proven pattern incl. the `end_at`
   gotchas documented in its module docstring; Daytona stop/start with
   `auto_stop_interval`).
2. **Mounted persistent workspace volume** — durable named volume per scope identity,
   mounted into every sandbox created for that scope.
   - Daytona: native Volumes (`daytona.volume.get(name, create=True)` +
     `VolumeMount(volume_id, mount_path)`). GA, free.
   - E2B: Volumes are PRIVATE BETA (`Volume.create` + `volume_mounts={path: volume}`).
     Attempt volume mount; on failure (no beta access) fall back to suspend-only
     persistence, set `persistence.warning`, and DO NOT fail the block.

## Migration surface (Daytona branch only)

Replace E2B in: `blocks/code_executor.py` (+ tests), `blocks/claude_code.py`,
`util/sandbox_files.py`, copilot sandbox tooling (`copilot/tools/e2b_sandbox.py`,
`copilot/sdk/e2b_file_tools.py`, `copilot/tools/bash_exec.py`) if feasible in scope,
settings (`Secrets.e2b_api_key` → keep, add `daytona_api_key`), provider registration,
env templates, frontend provider maps. Existing block IDs and schemas stay UNCHANGED
(drop-in backend swap; `sandbox_id` now carries a Daytona sandbox id).
`util/sandbox_files.py` couples to a 3-method surface (`commands.run`, `files.read`,
find-newer-than listing) — implement a Daytona adapter exposing that same surface.

Registration checklist for the new provider (from the dev-branch code):
- `ProviderName.DAYTONA = "daytona"` in `backend/integrations/providers.py`
- `"daytona": ("Sandboxed dev environments", ("api_key",))` in
  `backend/blocks/_static_provider_configs.py`
- `daytona_credentials` system credential (fixed id
  `7f3b9e2a-4c8d-4f1e-9a6b-2d5c8e7f0a3b`) in
  `backend/integrations/credentials_store.py` + `DEFAULT_CREDENTIALS` + conditional
  append gated on `settings.secrets.daytona_api_key`
- `daytona_api_key` in `backend/util/settings.py` `Secrets` + `backend/.env.default`
  (`DAYTONA_API_KEY=`)
- Frontend: `specialCases` in `src/providers/agent-credentials/helper.ts`
  (`daytona: "Daytona"`), `providerIcons` in
  `src/components/contextual/CredentialsInput/helpers.ts`, admin
  `DEFAULT_COST_PER_SECOND` map in `src/app/(platform)/admin/platform-costs/helpers.ts`

## Cost model (both branches)

1. **Platform credits** — add `BLOCK_COSTS` entries in
   `backend/data/block_cost_config.py` using the existing pattern
   (`BlockCostType.SECOND`, `cost_filter` on the system credential):
   - `CreateDesktopSandboxBlock`, `DesktopActionBlock`, `DesktopCommandBlock`,
     `DesktopFileBlock`, `StopDesktopSandboxBlock`: `cost_amount=1, cost_divisor=5`
     (1 credit / 5 s walltime — desktops are ~2× the vCPU of code sandboxes).
2. **Provider cost telemetry** (the comparison instrument) — every block emits
   `cost_meter`:
   ```json
   {
     "provider": "e2b" | "daytona",
     "sandbox_id": "...",
     "wall_time_s": 12.3,
     "resources": {"vcpu": 2, "ram_gib": 4},
     "estimated_cost_usd": 0.00123,
     "rate_usd_per_hour_running": 0.117,
     "rate_basis": "<short human-readable pricing formula>"
   }
   ```
   Rates live as constants in `blocks/desktop/_cost.py` (one file, provider-specific
   values per branch):
   - E2B: $0.000014/vCPU/s + $0.0000045/GiB-RAM/s (published per-second rates).
   - Daytona: published per-vCPU-hour + per-GiB-RAM-hour + per-GiB-disk-hour rates from
     https://www.daytona.io/docs/en/billing.md (record the exact numbers found there in
     `_cost.py` — Linux rates, not the Windows $0.0858/vCPU/h rate).
   Also record `merge_stats(NodeExecutionStats(...))` walltime so SECOND billing
   reconciles (this happens automatically via executor manager).

## Frontend (both branches, identical)

- New renderer `DesktopStreamRenderer` in
  `src/components/contextual/OutputRenderers/renderers/DesktopStreamRenderer.tsx`,
  registered in the registry `index.ts` with priority above LinkRenderer; `canRender`
  matches objects with `kind === "desktop_stream"`. Check the duplicated renderer tree
  under `NewAgentLibraryView/.../OutputRenderers/` and register there too if that surface
  needs it.
- Renders an "Interactive Desktop" panel: `<iframe src={url}>` sized 16:9,
  pointer-events enabled (it is interactive), `allow="clipboard-read; clipboard-write"`,
  fullscreen toggle, "Open in new tab" link, provider badge. Model on
  `HTMLRenderer.tsx` / the copilot `ArtifactReactPreview.tsx` iframe handling.
- Design system conventions per `frontend/CONTRIBUTING.md`; Vitest+RTL integration test
  for the renderer.

## Comparison harness

`docs/platform/sandbox-provider-comparison.md` (same file, both branches):
standard test graph (create desktop [user scope] → command `xdg-open` a page → actions:
screenshot, click, type → file write to workspace → stop[suspend] → recreate → verify
workspace file persisted → stop[destroy]), what to record per provider: cold-start s,
reconnect-after-suspend s, action round-trip s, stream first-paint s, $/h running
(from cost_meter), $/mo suspended, volume semantics notes, stream quality notes. Include
a results table template with an empty column per provider.
