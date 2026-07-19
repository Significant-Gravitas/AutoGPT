# Copilot chat: duplicate / flickering messages — problem & design

Status: design handoff (2026-07-18). Not yet implemented. Written for a coding
agent picking this up cold; all file references verified against branch
`pwuts/streamline-autopilot-agent-creation` unless marked _verify_.

## Symptoms

- Messages (especially tool outputs) render **twice** during a live turn;
  a page reload shows them once.
- Messages **flicker**: appear, disappear, reappear repeatedly during long
  turns.
- Reproduced concretely with the engine-switch feature (session
  `cd405f7f-8bef-46d5-b387-4ccc414527d5`): the `enter_agent_building_mode`
  tool output rendered twice live, once after reload. Backend was verified
  clean — exactly **1** persisted tool-result row, the tool-output event
  appears exactly **once** in the turn's Redis stream
  (`chat:stream:<turn_id>`, inspect via `XRANGE`), and the follow-up turn's
  stream replays nothing. The duplication is purely client-side.

## Root cause

The chat UI has **three sources of truth** for the same content, with no
shared identity or ordering contract, reconciled heuristically:

1. **Live SSE stream** — AI SDK v5 `useChat` +
   `DefaultChatTransport` pointed at `POST /api/chat/sessions/{id}/stream`
   (`frontend/src/app/(platform)/copilot/copilotStreamTransport.ts`).
   Parts append into the AI SDK's internal message state.
2. **Replay stream** — `GET /api/chat/sessions/{id}/stream` resumes the
   _active_ turn by replaying its Redis stream **from `0-0`**
   (backend: `subscribe_to_session`, `backend/api/.../routes.py` around
   line 1530-1565; events sourced from `chat:stream:<turn_id>` Redis
   streams via `backend/copilot/stream_registry.py`). Reconnect triggers:
   mount/reload, session switch, `useWakeResync`, watchdogs, and the
   post-finish poll in `useCopilotStream.ts` (`handleFinish` — recently
   changed to retry `FINISH_REFETCH_ATTEMPTS=4` times to catch
   server-initiated continuation turns). **Replay is unconditional from
   zero**: anything the client already rendered duplicates.
3. **DB refetch** — `useGetV2GetSession` →
   `convertChatSessionMessagesToUiMessages`
   (`frontend/src/app/(platform)/copilot/helpers/convertChatSessionToUiMessages.ts`).
   Because the backend **flushes messages to the DB mid-turn** (intermediate
   persistence in both SDK and baseline services), a refetch during a live
   turn returns a _partial snapshot_ of content the stream is concurrently
   delivering. Where refetch results _replace_ list state, streamed-but-not-
   yet-persisted content vanishes; the stream re-adds it; the next
   watchdog/wake refetch drops it again → the observed flicker loop.

Duplicate-render mechanism for the engine-switch case specifically: the
post-finish reconnect poll re-subscribed while the just-finished turn was
still the registered active stream → GET replayed that turn's events from
`0-0` into a chat that had already rendered them.

New event sources are multiplying (server-initiated continuation turns from
the engine switch, `schedule_followup`-fired turns, reconnect/watchdog/wake
paths), and each currently multiplies the reconciliation surface. Per-cause
guards (e.g. "only reconnect if the active turn differs from the one that
just finished") are symptom patches; the class of bug survives.

## Design: stable IDs + resume watermark + one merge policy

Standard event-sourcing treatment, applied at one choke point each on the
backend and frontend. After this, _every_ merge (initial load, refetch,
live append, replay, new server-initiated turns) is idempotent by
construction.

### 1. One identity everywhere (backend + converter audit)

Every renderable message must carry a **backend-minted ID that is identical
in the DB row, the live stream, and the replay**.

- Plumbing that already exists:
  - User rows already use the frontend's per-click UUID as the Prisma PK —
    explicitly documented as the atomic dedup primitive
    (`backend/copilot/model.py`, `ChatMessage.id` docstring).
  - Stream `start` events carry a `messageId`
    (`backend/copilot/response_model.py`, `StreamStart`).
  - DB rows have per-session monotonic `sequence` numbers.
- The work (_audit + align_): make assistant/tool/reasoning rows persist
  with the same IDs the stream announced (or add a deterministic mapping),
  and make `convertChatSessionMessagesToUiMessages` emit UI message IDs
  from row IDs so they collide (intentionally) with the AI SDK's streamed
  message IDs. Today the converter reads only
  id/role/content/tool_calls/sequence/duration_ms/created_at — verify what
  it uses as the UI id.

### 2. Watermark-based resume (kills the replay-duplication class)

- **Backend**: tag every SSE event with its Redis stream entry ID using the
  standard SSE `id:` field (the events already live in `XRANGE`-able
  streams, so entry IDs are free). Accept a `since=<entry-id>` query param
  (or the standard `Last-Event-ID` header) on the GET resume route and
  replay from there instead of `0-0`. ~Half day; Redis `XRANGE (since, +]`
  is the whole implementation.
- **Frontend**: in `copilotStreamTransport.ts`, track the highest applied
  entry ID per turn (a small per-session map). On reconnect, request
  `since=<watermark>`. Defensively drop any incoming event `<= watermark`
  that arrives anyway (server not yet updated, races). One shim, all
  replay duplicates gone deterministically — including the engine-switch
  double render, without its bespoke guard.

### 3. Merge policy: streams win above the watermark

Where DB-refetch results enter chat state (the hook that seeds/updates
`useChat` messages from `useGetV2GetSession`): refetch content may only
**upsert by ID at or below** the last persisted sequence it returns; it must
**never remove** messages the live stream added above that point. One rule,
enforced in one place, ends the refetch-vs-stream tug-of-war (the flicker).

## Implementation order & scope

1. **ID-alignment audit** (~half day, do first — determines whether the rest
   is assembly or surgery): trace one turn end-to-end (stream `messageId`s
   vs persisted row IDs vs converter output IDs) and list every mismatch.
2. **Backend SSE entry IDs + `since` param** (~half day).
3. **Transport watermark shim** (~half day).
4. **Refetch merge policy** (~1 day incl. tests; the AI SDK's message state
   is controllable via `setMessages` — verify the exact integration point in
   `useCopilotStream.ts`).

Total ≈ 2–2.5 days. Ship as its own PR (platform fix, independent value);
ideally lands **before** the engine-switch feature PR so server-initiated
continuation turns demo cleanly.

## Acceptance criteria

- Reconnect mid-turn (kill/restore network, wake from sleep, watchdog) never
  duplicates already-rendered parts; replay starts at the watermark.
- A DB refetch during an active stream never removes or reorders streamed
  content; post-reload rendering is byte-identical to pre-reload.
- A server-initiated continuation turn (engine switch, `schedule_followup`)
  appears exactly once, live, with no duplicate of the previous turn's
  content.
- Soak: run a long build session with aggressive `useWakeResync`/watchdog
  triggering (e.g. background the tab repeatedly) — zero duplicates or
  flicker.

## Non-goals / notes

- No change to the AI SDK version or wire protocol; the `id:` field and
  dedup shim are transport-layer additions.
- Backward compatible: an old frontend against a new backend ignores SSE
  `id:`; a new frontend against an old backend falls back to from-zero
  replay with client-side dropping (still deduped, just wasteful).
- The engine-switch reconnect guard ("different turn only") becomes
  unnecessary once the watermark lands; don't implement both.
