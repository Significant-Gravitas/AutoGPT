# Background subtasks and budgeted follow-ups

Design for [SECRT-2813](https://linear.app/autogpt/issue/SECRT-2813).
Nick chose **wake Otto for a budgeted follow-up** after the harness investigation.

## First increment

`run_sub_session(background=true)` accepts a fresh, queued subtask and returns
immediately. It is off behind `copilot-background-subtasks`, and foreground
behavior remains the default. The model should opt in only when the user asks
for background work. The foreground safeguard from #15127 stays enabled.

The original chat shows a background subtask card, current status, Stop, and a
link to its own chat. On completion, failure, cancellation, or timeout, Otto
wakes in the original chat for one summary. This follow-up cannot call tools
or start more work. It waits until the chat is idle, retaining the original
budget tree, cost attribution, credential pins, account route, and scope.
Opening the chat again recovers the messages; an already open chat refreshes
while the bounded background delivery is pending.

The child has ten minutes including queue time. The follow-up has two minutes
including queue time and never extends an inherited parent deadline. A busy
parent can postpone delivery for up to ten additional minutes; after that,
its next idle opportunity receives a fixed terminal notice instead of model
work. Subtask Stop cancels that child. Parent Stop cancels background work
launched by that turn and suppresses its automatic wakes. Ordinary completion
of the parent does not cancel background work.

## Durable ownership

`ChatBackgroundTask` stores the obligation, original execution context, outcome,
and delivery state. Its identity comes from the parent session, launching turn,
and actual tool-call ID. Accepting the obligation and writing the child's queued
message/status share a Postgres transaction. Accepting a follow-up and queuing
its message/status likewise share one transaction. Redis session locks serialize
these writes with chat writers; row locks and conditional updates fence retries.

A five-second scheduler pass recovers accepted work and undelivered outcomes.
Normal worker completion saves the child result before closing its stream. Each
stage has a durable execution claim: Rabbit redelivery cannot start the same
model work again. Interrupted work fails closed rather than replaying possible
side effects. Delivery retries reuse one stable message/turn ID.
If the outcome write failed but Redis still records successful completion for
the same owner and claimed child turn, recovery saves its original result.
Missing or mismatched completion records are not evidence of success.

The saved result is limited to the child's original turn, capped at 24,000
characters, and linked to its full chat. A follow-up is synthetic input, not a
message typed by the user. It is marked tainted and has an empty tool ceiling;
the UI renders a short labeled notice instead of the internal prompt. Existing
engine permission checks still enforce the ceiling.

Ownership, organization/team, expert, and account-route checks run before
acceptance, delivery, and execution. Deleted or changed-scope destinations do
not receive a result. The database worker's recovery listing is internal and
global; operations on individual tasks require the recorded owner.
If only the chat's connection changes before delivery, its next idle opportunity
gets one fixed notice linking to the subtask, without running a model follow-up.

Dispatch records the active turn identity in session metadata in Postgres.
If Redis stream metadata is lost, parent Stop cancels that turn's children and
suppresses their wakes in the same transaction that releases the parent slot.
Recovery checks the observed session timestamp under a row lock, so a stale
Stop cannot cancel a later turn's work.

## Budget semantics

The child and its one follow-up share one admitted background node in the
original tree. Queue promotion and worker pickup recheck that tree; there is no
new root or refreshed allowance. Both stages use the existing allowance gate
and provider-specific spend controls. A cancelled queued child retains its
node because its terminal follow-up may still consume model work.

The existing tree ledger meters settled spend. It does not reserve concurrent
spend and is not a strict no-overshoot dollar cap. This PR does not change that
accounting model. If the allowance or tree is exhausted, Otto does not get an
unbudgeted summary; the chat retains the terminal status and refusal/expiry
notice, with the detailed result available in the subtask.

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

Apply the additive migration, deploy compatible API, scheduler, and workers,
then enable the flag for test accounts. Keep the flag off during a mixed-worker
rollout. Disabling it stops new opt-ins; already accepted tasks still drain.

Regression coverage includes foreground behavior, post-turn delivery, a busy
parent, duplicate launches and workers, racing recovery passes, budget depletion
at promotion and pickup, cancellation, deadlines, and changed ownership/scope.
The worker-to-parent test uses real Postgres and Redis with scripted generation
and a captured queue publish. Frontend integration tests exercise status/Stop;
history and refresh tests cover labeled delivery after the original reply.
These checks do not substitute for a preview smoke test with a live provider.
