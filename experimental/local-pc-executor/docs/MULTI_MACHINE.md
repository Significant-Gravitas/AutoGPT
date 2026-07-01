# Multi-Machine Orchestration

> **Status**: Draft v0.1 — spec only; implementation aspirational (v2).
>
> Closes VISION.md §6. One copilot session coordinates work across
> multiple shims running on different machines: dev box + build server
> + NAS + Pi cluster. The single-shim case (one session ⇄ one shim) is
> what v1 ships; this doc pins how the wire + platform expand to N
> shims so the v1 design doesn't paint us into a 1:1 corner.

The simplest mental model: today `session_id` is unique per copilot
chat session and routes to one shim via `ShimConnectionManager`. In
multi-machine v2, the same `session_id` can have N attached shims,
each identified by `machine_id`. Tool calls carry an optional
`target_machine` hint; absent the hint the platform picks via routing
policy.

---

## Wire changes

### `MACHINE_LIST_REQUEST` (platform → all shims OR a specific shim) — NEW v2 op

Asks the shim for its current `machine_id` + capability snapshot, plus
whatever the platform-side registry knows about siblings. Issued by
the platform's `LocalPCRouter` on the first multi-shim turn so Claude
can `list_machines()`.

```json
{
  "type": "MACHINE_LIST_REQUEST",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {}
}
```

The shim responds with its own metadata; the platform synthesizes
the union across all attached shims for the same session.

### `MACHINE_LIST_RESPONSE` (shim → platform)

```json
{
  "type": "MACHINE_LIST_RESPONSE",
  "id": "req-uuid",
  "ts": 1234567890.1,
  "payload": {
    "machine_id": "this-shim-uuid",
    "hostname": "alices-laptop",
    "platform": "darwin",
    "arch": "arm64",
    "capabilities": ["shell", "files", "computer_use"],
    "hardware_devices": []
  }
}
```

### Every request envelope grows an optional `target_machine` field

```json
{
  "type": "FILE_READ",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "version": "2.0",
  "target_machine": "alices-laptop",   // null = router decides
  "payload": { "path": "/Users/alice/notes.md", ... }
}
```

When `target_machine` is null and N > 1 shims are attached, the
platform-side `LocalPCRouter` falls back to a routing policy (see
below) and stamps the chosen `machine_id` on the wire envelope before
forwarding. The shim never has to know it was chosen — it just
executes.

### Cross-shim file transfer

Common ask: "Read `/Users/alice/data.csv` on laptop, write to
`/srv/incoming/data.csv` on the build server." The MVP shape: platform
streams the read response through itself and into the write request.
Adequate for tens-of-MB files; an optional v3 enhancement is a
**direct shim-to-shim tunnel** that bypasses the platform for bulk
transfers.

The wire op the platform issues is a synthesized pair — there's no
single `FILE_COPY` wire message. Same goes for executing the output of
machine A's command as input to machine B's: the platform glues two
single-machine ops.

---

## Platform-side router

`autogpt_platform/backend/backend/copilot/tools/local_pc_router.py`
(new module, v2).

```python
class LocalPCRouter:
    """Picks a shim for a tool call when N > 1 are attached to a session.

    Routing policy precedence:
      1. explicit `target_machine` in the tool call → use that.
      2. capability requirement (e.g. EXECUTE_COMMAND needing `shell`
         on a host with the file in question already → bias toward
         that machine).
      3. user-configured default machine (per-session preference saved
         to Redis copilot:localpc:default_machine:{user_id}).
      4. round-robin across all attached shims for the session.
    """

    async def pick(
        self, session_id: str, op: WireOp, *, hint: str | None = None
    ) -> str:
        ...
```

`ShimConnectionManager` grows a `dict[session_id, dict[machine_id, ws]]`
instead of `dict[session_id, ws]`. The new tool `list_machines` (an
MCP tool the platform exposes) returns the per-machine snapshots so
Claude can reason about which machine to target.

---

## Claude-facing tool augmentation

A new MCP tool exposed when N ≥ 2 shims are attached:

`list_machines() -> list[MachineSnapshot]` where MachineSnapshot is
the `MACHINE_LIST_RESPONSE` payload shape plus a `last_active_ts`.

The existing `local_pc_*` tools (screenshot, click, execute_command,
file_read, etc.) all gain an optional `target_machine: str | null`
parameter that maps onto the wire envelope's field. Unset = router
decides; set = override.

The platform's tool descriptions auto-include the routing semantics
when N ≥ 2 ("If you don't pass target_machine, the router picks based
on capability + recency.") so the LLM knows when targeting matters
vs. when it's safe to omit.

---

## Multi-machine session lifecycle

- **First connection (N = 1)**: indistinguishable from v1. Tool calls
  go to the single shim; `target_machine` always omitted.
- **Second connection (N = 2)**: platform sends `MACHINE_LIST_REQUEST`
  to both, caches the responses, and starts including `target_machine`
  on outbound envelopes. The first shim has no idea the topology
  changed — its existing in-flight ops continue against its own
  `machine_id`.
- **Shim disconnect mid-session (N ⇒ N-1)**: pending ops against the
  dead `machine_id` raise `MACHINE_UNREACHABLE` (new error code). The
  LLM sees the translation and either retries on a different machine
  or asks the user to bring the machine back.
- **Same machine re-attach (re-HELLO with same `machine_id`)**: per
  the Q2 session-ownership rules locked in #36, this is a re-connect
  / takeover, not a new attachment. `ShimConnectionManager` replaces
  the old ws + keeps the same `machine_id` slot in the session.
- **Different-machine reattach with conflicting `target_machine`**: the
  router refuses with `MACHINE_NOT_ATTACHED`. The platform never
  silently picks a "close enough" substitute machine.

---

## Permission model

Each shim authenticates independently against the same OAuth
application (the user runs `autogpt-shim auth` on each machine).
Sessions are bound to `(user_id, session_id)`, not `(user_id,
machine_id, session_id)` — so a machine the user has authed and
connected can join any of their existing sessions.

This means: a user with two laptops both running the shim, joining
the same session_id, get N = 2 attached. The router treats them as
peers. **There's no "primary" machine and no implicit hierarchy.**

Per-machine capability gating still works as before: HELLO declares
what the machine can do, and the router refuses ops on machines that
lack the requested capability with `CAPABILITY_NOT_GRANTED`.

---

## New error codes (v2)

- `MACHINE_NOT_ATTACHED` — `target_machine` specified but no shim with
  that `machine_id` is currently attached to this session.
- `MACHINE_UNREACHABLE` — shim was attached but disconnected before
  the op completed. Caller can retry on another machine via the same
  router, or wait and retry on this one.
- `MACHINE_AMBIGUOUS` — multiple shims match a partial machine
  identifier (e.g. user passed `target_machine: "laptop"` and two
  hostnames start with "laptop"). Router refuses; caller picks one
  unambiguously.

---

## v2 vs v3 split

**v2** (when first customer asks for it):
- N attached shims per session
- `target_machine` on every wire envelope
- `list_machines` MCP tool
- Round-robin + capability-based routing
- Cross-shim file transfers via platform-mediated read-then-write

**v3** (only if v2 produces real demand):
- Direct shim-to-shim tunnel for large file transfers (avoids the
  platform bandwidth tax)
- Cross-machine pipeline ops (run command A on machine X, pipe stdout
  into command B on machine Y)
- Distributed work — split a Claude turn across multiple machines that
  each see only their own slice of the conversation
- Cluster-aware policies (CPU-bound work to the workstation, IO-bound
  to the NAS, etc.)

v3 is well past the holy-grail line. Probably never ships unless the
product pivots.

---

## What this doc deliberately does NOT pin

- **Routing policy weights** beyond the precedence list above —
  implementation details that should be tunable per deployment.
- **Multi-tenant** ("can my organization share a shim with another
  user?") — explicit no in v1 per the threat model in SECURITY.md;
  v2 multi-machine doesn't relax it.
- **Cross-machine permissions inheritance** (a tool call originated on
  machine A but writes a file on machine B — who owns the audit
  trail?) — answered at v2 implementation time; the AUDIT_LOG.md
  format already supports `details.target_machine` if needed.

---

## References

- VISION.md §6 — the original capability description.
- PROTOCOL.md — envelope, version negotiation (#35), session
  ownership (#36).
- SECURITY.md — single-tenant invariant.
- AUDIT_LOG.md — per-record `details.machine_id` if v2 needs it.
