# Privacy Mode

> **Status**: Draft v0.1 — spec only; implementation aspirational (v2).
>
> Closes VISION.md §5. Files never leave the user's machine; a local
> LLM (Ollama / llama.cpp / LM Studio) does the content work; only
> the *answer* — not the file body — flows back to the cloud copilot.
> Big design surface; this doc pins the contract so v1 doesn't make
> a choice that forecloses it.

The threat model privacy mode solves: a user wants Claude (in the
cloud) to reason about a medical record / private codebase / financial
file / NDA-bound document. The file shouldn't transit the AutoGPT
platform at all, but Claude's planning still happens cloud-side. So:
the platform sends an abstract task ("summarize this file" / "find the
TODO comments" / "extract dates"), the shim resolves the file locally,
the local LLM does the content-touching work, and only the summary /
list of TODOs / dates come back as the tool result.

---

## What changes vs the default flow

**Default flow today:**
1. Platform calls `FILE_READ(path)` over the wire.
2. Shim reads bytes, sends `FILE_CONTENTS(content)` back.
3. Platform forwards the content into Claude's context.
4. Claude reasons over the raw content.

**Privacy mode:**
1. Platform calls `FILE_READ(path, mode="stub")` over the wire.
2. Shim returns a `FileStub` — `{path, size, mime_type, sha256,
   schema_preview}`. No content.
3. Claude sees the stub and decides what it wants to know about the
   file ("summarize" / "extract dates" / "is this a valid CSV").
4. Platform sends a `LOCAL_LLM_TURN` wire op carrying the task
   description + the stubbed file reference.
5. Shim resolves the file locally, runs it through the local LLM with
   the task as the prompt, returns only the LLM's output.
6. Claude sees the *output*, not the content.

The platform never sees the raw bytes; the cloud LLM never sees the
raw bytes; only the shim and the local LLM ever touch them.

---

## Wire ops

### `FILE_READ` (extended) — `mode` field

Existing `FILE_READ` payload grows `mode: "content" | "stub"`. Default
`"content"` preserves today's behavior.

```json
{
  "type": "FILE_READ",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "path": "/Users/alice/medical-records.pdf",
    "mode": "stub",
    "schema_preview_bytes": 256    // optional; first N bytes for type-detection only
  }
}
```

### `FILE_CONTENTS` (response) — stub branch

When `mode == "stub"`, the response carries no content:

```json
{
  "type": "FILE_CONTENTS",
  "id": "req-uuid",
  "ts": 1234567890.1,
  "payload": {
    "mode": "stub",
    "path": "/Users/alice/medical-records.pdf",
    "size_bytes": 482_310,
    "mime_type": "application/pdf",
    "sha256": "9c81f4a2…",          // content fingerprint
    "schema_preview": "%PDF-1.7\n...",  // first schema_preview_bytes, base64-encoded
    "stub_id": "stub_<uuid>"        // opaque handle for LOCAL_LLM_TURN reference
  }
}
```

`stub_id` is the same shim-minted-UUID pattern locked in Q2: never
reused, wiped on HELLO. Subsequent `LOCAL_LLM_TURN` ops reference the
stub_id rather than re-sending the path (so the file doesn't get
re-resolved + re-fingerprinted each time, and a TOCTOU change between
turns surfaces as `STUB_STALE` if the sha256 has drifted).

### `LOCAL_LLM_TURN` (platform → shim) — NEW v2 op

```json
{
  "type": "LOCAL_LLM_TURN",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "model": "llama3.2:3b",        // must be in HELLO.local_llm_models
    "task": "Summarize this document in 3 bullet points.",
    "stub_refs": ["stub_<uuid>"],  // 1 or more FileStubs to feed into the prompt
    "context": "The user is reviewing their Q3 expenses.",  // optional
    "max_tokens": 1024,
    "temperature": 0.0,
    "verify_sha256": true          // re-fingerprint and refuse if drifted from stub
  }
}
```

The shim:
1. Looks up each `stub_id` in its per-process stub registry; refuses
   unknown / stale with `STUB_STALE`.
2. If `verify_sha256`, re-hashes the file; refuses if drifted.
3. Resolves stubs to actual file content (still local — never leaves
   the machine).
4. Constructs the local-LLM prompt: `{system: <task>, user:
   {context}, file_attachments: [<content>]}`. Exact prompt shape
   depends on the local backend (Ollama chat vs llama.cpp vs LM
   Studio); the shim normalizes.
5. Streams the local-LLM response back as `LOCAL_LLM_TURN_RESPONSE`
   chunks (same pattern as the platform's own SSE streaming).
6. Audit-logs the op as `LOCAL_LLM_TURN` with `{model, stub_ids,
   task_length, output_length, duration_ms}` — never the raw content
   or output text. Per AUDIT_LOG.md "what's never logged."

### `LOCAL_LLM_TURN_RESPONSE` (shim → platform)

```json
{
  "type": "LOCAL_LLM_TURN_RESPONSE",
  "id": "req-uuid",
  "ts": 1234567890.5,
  "payload": {
    "output": "The user spent $2,140 on…",
    "finish_reason": "stop",
    "tokens_used": 384,
    "duration_seconds": 4.2,
    "model": "llama3.2:3b"
  }
}
```

For streaming, the shim emits multiple intermediate `LOCAL_LLM_TURN_CHUNK`
frames with `{delta: "..."}` payloads, then a final
`LOCAL_LLM_TURN_RESPONSE` with the assembled output.

### `STUB_STALE` (new error code)

The `stub_id` doesn't exist, was wiped on HELLO, or `verify_sha256` is
true and the file's sha256 has drifted since the stub was minted.
Caller re-issues `FILE_READ(mode="stub")` to mint a fresh one.

---

## Platform-side router

`autogpt_platform/backend/backend/copilot/tools/privacy_router.py`
(new module, v2). Wires into `e2b_file_tools.py` (the existing read
chokepoint).

```python
class PrivacyRouter:
    """When privacy mode is on, FILE_READ tool calls return a FileStub
    rather than content. A follow-up LOCAL_LLM_TURN consumes the stub.

    Activation gate (any one is enough):
      - config.privacy_mode_default = True (org-level / always on)
      - LD flag privacy-mode-per-user enabled for this user
      - per-session opt-in via a new copilot UI toggle
      - per-tool-call opt-in via `read_file(..., privacy=True)`
    """
```

The router also strips file-content references from the conversation
history that flows to Claude. The cloud LLM only ever sees
`{stub_id, sha256, size, mime_type, schema_preview}` for privacy-mode
files — never the file body itself.

---

## Claude-facing tool augmentation

Two new MCP tools (gated on privacy-mode active):

- `read_file_stub(path)` → returns the stub payload. Claude reads
  this to decide what local-LLM task to fire.
- `local_llm_query(stub_ids, task, model?)` → fires `LOCAL_LLM_TURN`
  and returns the output. Errors surface via the existing
  local_pc_errors translator with privacy-aware messages ("the local
  LLM couldn't complete this task; consider reading the file
  unencrypted with `read_file` if privacy mode allows").

The existing `read_file` MCP tool, in privacy mode, refuses
content reads with a structured `PRIVACY_MODE_BLOCKED` error and
recommends the stub + local_llm_query pair.

---

## What about non-file content?

`EXECUTE_COMMAND` stdout is the obvious next surface. v2 doesn't
extend privacy mode there — shell output is too unstructured to
reliably stub. Document this gap: "if you don't want command output to
reach the cloud, run the command locally yourself; don't ask Claude
to run it."

Screenshots in computer-use mode are also content. v2 leaves
screenshots as-is — they flow to the cloud LLM unchanged. v3 could
add a "local OCR + local vision model" path mirroring this pattern,
but the local-vision-model maturity isn't there yet (as of 2026-06).

---

## Audit log

Privacy mode adds two new `op` values to AUDIT_LOG.md:

- `FILE_READ_STUB` — stub mint. `details = {path, sha256, size_bytes,
  mode: "stub"}`.
- `LOCAL_LLM_TURN` — local-LLM execution. `details = {model,
  stub_ids, task_length, output_length}` — never the task text, never
  the output text. The user reviews the audit log to verify "Claude
  asked the local LLM N times about file X, output was K tokens
  long" without leaking what was actually asked / answered.

This is intentional: the audit log itself is per the same privacy
posture as the wire — sees metadata, never bodies. A user paranoid
enough to want privacy mode is paranoid enough not to want a tamper-
evident log of every question they asked their own local model.

---

## v2 vs v3 split

**v2** (when first customer asks):
- `FILE_READ.mode=stub` + `FileStub`
- `LOCAL_LLM_TURN` + streaming response
- `PrivacyRouter` activation gates
- `read_file_stub` + `local_llm_query` MCP tools
- Audit log entries

**v3** (only if v2 demand):
- Privacy mode for `EXECUTE_COMMAND` stdout (a local-LLM
  summarization pass before stdout leaves the shim)
- Privacy mode for screenshots (local OCR + local vision)
- Encrypted file references — when the platform DOES need to know
  a path identity across turns without seeing it, swap path for
  encrypted alias

---

## Dependencies

- **Local LLM routing (#33)** must land first. Privacy mode is
  fundamentally "route this turn to the local LLM" with the wrinkle
  that the file content doesn't transit the cloud. Most of the wire +
  router work is shared.
- **Wire protocol version (#35)** must land first. Privacy-mode wire
  ops (`mode=stub`, `LOCAL_LLM_TURN`) are forward-compatible additions
  to the same major version, but the version negotiation needs to
  exist for the platform to know the shim supports them.

---

## References

- VISION.md §5 — the original capability description.
- AUDIT_LOG.md — "What's never logged" + new op values added here.
- PROTOCOL.md — `FILE_READ` envelope (extended), new LOCAL_LLM_TURN
  ops, STUB_STALE error code.
- MULTI_MACHINE.md — privacy mode + multi-machine compose: in a v2 +
  v3 world, privacy-mode stubs are minted on the machine that holds
  the file, and `LOCAL_LLM_TURN` is routed to the same machine.
