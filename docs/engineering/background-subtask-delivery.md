# Background subtasks that return to the chat

Design for [SECRT-2813](https://linear.app/autogpt/issue/SECRT-2813).
The first increment (#15733) woke Otto for a bounded, tool-free summary. Nick
then chose **real background work**: the subtask runs until it is done, and the
chat carries on from the result with everything it had, with no clock of its own.

## Background work

`run_sub_session(background=true)` accepts a fresh, queued subtask and returns
immediately. It is off behind `copilot-background-subtasks`, and foreground
behavior remains the default. The model should opt in only when the user asks
for background work. The foreground safeguard from #15127 stays enabled.

The original chat shows a background subtask card, current status, Stop, and a
link to its own chat. The child runs under the same spawn rules as a foreground
subtask: the launching turn's tools minus the descent denials, its budget tree,
cost attribution, credential pins, account route and scope, and it may spawn
further work within the tree's depth. It keeps no deadline of its own; the only
deadline it can have is one the launching turn already had.

When the child completes, fails or is stopped, Otto is woken once in the
original chat, as soon as that chat is idle, however long that takes. The wake
runs as a continuation of the launching turn: the same envelope, tools and
spend ledger, so the chat can read the result, keep working, or start more
subtasks. The wake message names the subtask and its outcome and points at
`get_sub_session_result`; what the subtask produced stays in its own chat and
reaches the model as a tool result, the way a foreground subtask's does, rather
than as text in a user message. The UI renders the wake as a short labeled
notice, not the internal prompt.

Subtask Stop cancels that child. Parent Stop cancels background work launched
by that turn and suppresses its wake. Ordinary completion of the parent does not
cancel background work.

## Durable ownership

`ChatBackgroundTask` stores the obligation, original execution context, outcome
and delivery state. Its identity comes from the parent session, launching turn
and actual tool-call ID. Accepting the obligation and writing the child's queued
message/status share a Postgres transaction. Accepting a continuation and
queuing its message/status likewise share one transaction. Redis session locks
serialize these writes with chat writers; row locks and conditional updates
fence retries.

A five-second scheduler pass recovers accepted work and undelivered outcomes.
Normal worker completion records the child's outcome before closing its stream.
Each stage has a durable execution claim: Rabbit redelivery cannot start the
same model work again. If the outcome write failed but Redis still records a
successful completion for the same owner and claimed turn, recovery records the
success; the same applies to a continuation that answered and only failed to
record that. Missing or mismatched completion records are not evidence of
success. A child whose chat went idle without an outcome is recorded as
interrupted, and the chat carries on from that; a child whose worker died is
released by the existing orphan handling when its chat is next read, which the
parent's card does while the work is pending.

Ownership, organization/team, expert and account-route checks run before
acceptance, delivery and execution. Deleted or changed-scope destinations do
not receive a result. If only the chat's connection changes before delivery,
its next idle opportunity gets one fixed notice linking to the subtask, without
a model turn.

Dispatch records the active turn identity in session metadata in Postgres, and
admission clears it, so one admission records one turn. If Redis stream
metadata is lost, parent Stop cancels that turn's children and suppresses their
wakes in the same transaction that releases the parent slot. Recovery checks
the observed session timestamp under a row lock, so a stale Stop cannot cancel
a later turn's work. A scheduler stop names the turn it stops, and the worker
ignores a stop whose turn is no longer the one running. The session endpoint
reports `background_outcome` for a child chat and `background_pending` for a
parent chat from the task record, so the cards and the parent's polling do not
depend on the launch or acceptance rows staying within the transcript page.

## Budget semantics

Time is not a budget. The child is a node of the launching turn's tree and is
metered by it like any spawned turn; a cancelled or refused queued child gives
its node back. The continuation runs under the launching turn's own envelope,
so it is governed by the user's allowance and spend limits the way every turn
of that chat is, not by the subtask's node. Queue promotion and worker pickup
recheck those limits.

The existing ledgers meter settled spend. They do not reserve concurrent spend
and are not a strict no-overshoot dollar cap. If the allowance is exhausted, the
continuation is refused with the usual limit notice, and the result stays
readable in the subtask's chat.

## Harness investigation and local reproduction

The locked Python Claude SDK is 0.2.161. Scripted local Anthropic responses and
the installed Claude Code 2.1.303 CLI reproduced the original failure: enabling
native background work, then closing the client at the first `ResultMessage`,
loses a helper that completes later. The existing foreground regression passes.
These tests used fake credentials, not live model calls.

Keeping `receive_messages()` open instead produced an initial result at 0.78s,
child completion at 12.80s, and another parent result at 12.88s, without another
user query. A stop probe received a terminal task update with `status=killed`.
A budget probe ended with `error_max_budget_usd`; its cumulative cost exceeded
the limit before stopping, demonstrating a soft limit rather than reservation.
See the [Claude SDK lifecycle](https://code.claude.com/docs/en/agent-sdk/subagents).

Native Codex's completion watcher uses `trigger_turn=false` in its newer path
and injects a notification in the legacy path ([source at e05028e7](https://github.com/openai/codex/blob/e05028e7b0fdd6a483a720920594b65f0b85a872/codex-rs/core/src/agent/control.rs#L453-L578)).
T3's delegated-task tool contract explicitly wakes the parent. No paid T3
delegation was needed for this investigation.

AutoGPT's Codex route uses the Claude SDK tool loop with HTTP inference, not a
native Codex process. Its baseline engine has its own tool loop. Microsoft 365
Copilot uses a provider chat API without these platform tools. Durable platform
queue ownership therefore fits both supported tool loops without depending on
a native harness's completion policy.

## Rollout and verification

Apply the migrations, deploy compatible API, database manager, scheduler and
workers, then enable the flag for test accounts. Keep the flag off during a
mixed-worker rollout. Disabling it stops new opt-ins; already accepted tasks
still drain.

Regression coverage includes foreground behavior, post-turn delivery, a busy
parent, duplicate launches and workers, racing recovery passes, budget depletion
at promotion and pickup, cancellation, an inherited deadline, and changed
ownership/scope. The worker-to-parent test uses real Postgres and Redis with
scripted generation and a captured queue publish. Frontend integration tests
exercise status/Stop; history and refresh tests cover labeled delivery after
the original reply. These checks do not substitute for a preview smoke test
with a live provider.
