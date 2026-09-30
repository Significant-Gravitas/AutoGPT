# Agent collaboration architecture

When AutoPilot works on a request, it can hand part of the job to another
agent — a teammate expert, or a fresh copy of itself with a clean context.
That agent can hand off part of *its* job in turn. This document describes
the limits that apply to everything started that way: which tools a
handed-off job may use, how deep the chain may go, how many agents one
request may use, and how much they may spend between them.

The short version: **a spawned agent can never do more than the agent that
spawned it, and everything one request starts shares one budget and one
head-count.** Both limits are enforced in one place in the code, and a job
that would exceed either is refused before it starts, with a message the
calling agent can act on.

The implementation is `autogpt_platform/backend/backend/copilot/tree.py`.

---

## A worked example

A user asks AutoPilot to draft a customer reply. AutoPilot delegates the
research to an Ops expert and the tone pass to a Growth expert:

```
user types  ──▶  AutoPilot turn                 ← the root; one tree opens here
                   ├──▶ delegate → Ops          ← depth 1
                   └──▶ delegate → Growth       ← depth 1
                          └──▶ sub-session      ← depth 2
```

Four agents, one tree. Ops and Growth each get every tool AutoPilot has
except the ones no delegated job should hold — connecting an integration,
deleting things, hiring another expert. The sub-session Growth starts is
narrower still, because it runs under Growth's own identity and shares its
memory. All four turns draw on one spend ceiling and one allowance of eight
agents: a fourth hop of depth would be refused, so would a ninth agent, and so
would any turn starting after the ceiling is reached.

None of that is visible to the user unless a limit is hit, and then what
they see is the agent finishing with what it has rather than an error.

---

## The vocabulary

Five words carry the rest of the document, plus two for the two
shapes a spawn can take.

A **turn** is one agent doing one round of work: it reads a request, calls
tools, and produces a response. A copilot session is a series of turns.

A **root** turn is one that no other agent started — a human typing, a
schedule firing, an `AutoPilotBlock` inside a graph, a message arriving from
a chat platform.

A **child** turn is one that a tool started: `run_sub_session`,
`delegate_to_expert` or `handoff_to_expert`. Every child has exactly one
parent.

A **tree** is a root turn plus every turn descended from it. The tree is the
unit that gets a budget and a head-count. It is keyed on the root turn, not
on the session, so a long-running expert thread opens a fresh tree with a
fresh budget on every human message.

An **envelope** is the set of limits one turn carries: which tree it belongs
to, how deep it sits, which tools it may call, whether its input is
untrusted, and when it must stop. Each turn has its own, and a child's is
computed from its parent's.

The two shapes get different treatment throughout:

- A **delegate** runs as a *different* expert — its own identity, its own
  memory, its own voice. `delegate_to_expert` and `handoff_to_expert` create
  delegates.
- An **isolate** runs as the *same* identity as its parent, with a clean
  context. `run_sub_session` creates isolates. Because an isolate shares its
  parent's memory and name, anything it stores the parent reads back next
  turn as its own, and anything it posts goes out under the parent's name.

---

## What a child may do

A child's tool set is computed from its parent's, and the computation can
only take tools away:

```
child.tools = (parent's tools ∩ what the spawn asked for) − denied
```

`derive_child_envelope` (`tree.py:187`) is the only function that produces
one. Nothing a model writes reaches it: the parent's own set comes from the
turn that is running, not from a database row, and the subtraction of the
denied list happens last, so no request shape can route around it.

**Every child loses these**, whatever kind it is (`DESCENT_DENIED_TOOLS`,
`tree.py:72`):

| tool | why |
|---|---|
| `connect_integration` | binds the user's account to a third party |
| `setup_agent_webhook_trigger` | registers a trigger that outlives the request |
| `update_preset` | edits saved configuration |
| `delete_folder`, `delete_preset`, `delete_schedule`, `delete_skill`, `delete_workspace_file` | irreversible |
| `hire_expert`, `raise_expert`, `update_expert`, `confirm_expert_change` | staffs the user's team |

None of these is part of a delegated job, so withholding them breaks nothing
that worked, and it removes reach a spawned agent should never quietly have.

**An isolate additionally loses these** (`ISOLATE_DENIED_TOOLS`,
`tree.py:96`): `post_to_chat_platform`, `schedule_followup`, `run_mcp_tool`,
`store_skill`, `memory_store`, `add_understanding`.

The asymmetry is the point. A delegate keeps all six, because it runs under
its own identity and its own memory — storing what it learned and posting a
scheduled update are its job. An isolate is scratch space under someone
else's name, so a memory it writes becomes the parent's memory and a message
it posts appears to come from the parent.

**There is no way to grant a denied tool back.** The subtraction is last and
absolute, so a parent that holds `post_to_chat_platform` still cannot pass it
to an isolate. A job that genuinely needs a denied tool is one a human
approves, not one an agent grants itself.

Two enforcement points, and the difference matters. The envelope's tool set
is turned into a permission filter so the model is not *shown* tools it may
not call, and `BaseTool.execute` (`tools/base.py:232`) *refuses* a call to
anything outside the envelope. Hiding is presentation; the refusal is the
control, and it covers both engines and every registered tool.

---

## How deep and how wide a tree gets

Depth is capped at **3 hops** (`MAX_DEPTH`, `tree.py:59`) and counts every
spawn kind, isolates included. A fourth hop is refused with "This task is
already 3 hops deep; do as much as you can yourself instead of passing it on
again."

Head-count is capped at **8 turns per tree, the root included**
(`tree_max_nodes`, `config.py:467`), so one request may start seven agents.
This is a lifetime budget for the request, not a concurrency limit — the
refusal says so, because an agent that reads "limit" assumes waiting will
free a slot, and here it will not.

Both are enforced at `dispatch_turn` (`executor/utils.py`), which is the one
function every turn passes through: the HTTP chat route, the scheduler,
`AutoPilotBlock`, and all three spawn tools. A turn that is refused there
never has any side effect — no session row, no queue message.

---

## What a tree can spend

A tree's spend ceiling is set when its first spawn opens it, from the user's
subscription plan:

```
ceiling = min( the user's remaining daily/weekly budget,
               max( ½ × the plan's daily allowance, $0.50 ),
               $10.00 )
```

`resolve_root_ceiling_microdollars` (`tree.py:380`) computes it; the three
constants are `tree_ceiling_fraction_of_daily`,
`tree_ceiling_floor_microdollars` and `tree_ceiling_microdollars`
(`config.py:442-466`). Scaling off the plan rather than a flat number is what
keeps the ceiling proportionate, and it is why a user with no subscription
gets zero: a plan that may not spend may not delegate.

On the code defaults, which is what production serves today — the two
LaunchDarkly flags that can override the base limits and the tier multipliers
(`copilot-cost-limits`, `copilot-tier-multipliers`) are both off there:

| Plan | Daily allowance | Per-tree ceiling | Per-turn cap |
|---|---:|---:|---:|
| NO_TIER | $0.00 | $0.00 | — |
| BASIC | $1.00 | $0.50 | $1.00 |
| PRO | $5.00 | $2.50 | $5.00 |
| MAX | $20.00 | $10.00 | $10.00 |
| BUSINESS | $60.00 | $10.00 | $10.00 |
| ENTERPRISE | $60.00 | $10.00 | $10.00 |

Allowances are the `ChatConfig` base of $1.00/day times the plan's tier
multiplier (`rate_limit.py:108`). The per-turn cap is a separate, older
control — `min($10.00, the user's remaining budget)`, floored at $0.50
(`sdk/service.py:255`) — that bounds a single turn on the Claude/OpenRouter
path. Ceilings are what a user who has spent nothing yet gets; they shrink
through the day, because remaining budget is the first term of the `min`.

**Roots never touch the ledger.** A tree only exists once something is
spawned, so the HTTP route, the scheduler and `AutoPilotBlock` are unchanged
by any of this and pay nothing for it.

### Spend is counted, not reserved

The ledger is a Redis hash per tree — `copilot:tree:<tree_id>`, holding
`ceiling`, `spent`, `nodes` and `max_nodes` — and `spent` only advances when
a turn *ends*, from the same code path that feeds the per-user rate limit
(`token_tracking.py:236`).

The consequence: **turns admitted at the same moment do not see each other's
cost**, so a wide fan-out can overshoot. Seven children dispatched in one
round all read `spent` before any of them has charged anything, and all seven
are admitted. The worst case is therefore seven turns each running to its own
per-turn cap — $7.00 on BASIC, $70.00 on MAX and above — not one turn's
overshoot. The head-count cap is what makes that finite, which is why the two
limits are enforced together and why eight is a small number.

Reserving at admission and reconciling at charge would fix it. It needs a
per-query spend figure the Codex transport does not report, so it is not
possible today: the SDK's own `max_budget_usd` is a per-query stop that
rounds upward and does not exist on that transport at all.

### When the ledger is unreachable

Spawns fail closed and roots are unaffected. If Redis cannot be reached, a
child turn is refused with "Could not account for this work right now; try
again shortly"; the user's own turn runs, exactly as it does when the
per-user rate limit degrades.

---

## Who may talk to whom

A child has exactly two edges: the request coming in from its parent, and
its report going back. Nothing else can address it, and it can address
nothing but its parent and its own children. Siblings never talk; if two
children need the same data, the parent gives it to both.

There are no sibling, peer or broadcast edges, and that is a choice rather
than an oversight. Two agents that can each wait on the other can deadlock,
and the sub-session resume path could form exactly that shape before the
creator rule below closed it. Every lateral edge is also a second place to
enforce a limit the parent already enforces once.

`handoff_to_expert` is the one edge that moves ownership: the handoff turn
is the last turn of the parent's tree, and every human message typed into the
handed-off thread afterwards is a new root with a new tree and a new ceiling.

Three rules keep that shape honest:

- **Only the creator may resume, poll or cancel a child.** Each spawn writes
  `delegated_by_session_id`, and the resume path requires it to match. A
  session cannot resume itself and siblings cannot resume each other, which
  removes a self- and mutual-deadlock. Reading is deliberately looser — it
  keeps its existing scope-equality rule, because a same-scope reader is the
  same identity under the same user, and narrowing it broke a sub reporting
  progress and a handed-off expert reading its own task.
- **A spawn never queues into a turn already in flight.** That path appends
  the message to the running turn's buffer, which would execute a child's
  prompt inside another session's turn under *that* turn's permissions. All
  three spawn tools pass `allow_queue=False`, as does `AutoPilotBlock`; a busy
  target is refused and the caller is told to wait or start fresh.
- **A question goes to the parent, not to the human.** Only a root has a
  human attached, so escalation travels the same edges as reports, in the
  same direction, and lands once.

---

## What inherits, and what does not

Three classes, and which one a parameter falls into decides its rule.

**Ceilings** can only narrow on the way down: the tool set, depth, the tree's
spend and head-count, the taint bit, the deadline, and the tenancy and
billing route (which must be identical, never merely narrower). A request
that asks to change one of these is clamped or refused, never granted.

**Identity-bound** state comes from *which* expert the child is, never from
its parent: the expert's soul and voice, its memory namespace, its
integrations and workflows, and its own weekly budget. A delegate writes to
its own memory namespace, not its parent's — memory crossing between
teammates is bounded by who was delegated to, not by the tree.

**Free within the ceiling** is the caller's choice: model, thinking effort,
how long to wait for a result. Model tier deliberately does not inherit. A
child that needs a stronger model for its narrow job should have one; what
must not widen is the spend, and the ledger bounds that whatever model runs.
Making tier monotone would push callers to over-provision the root so their
children could be capable, which raises spend — the thing being bounded.

The expert's own weekly budget stays separate from the tree ledger on
purpose. It is a per-identity guardrail its owner configured ("this expert
may not burn more than X a week on schedules"); the ledger is a per-request
spend bound. A breach pauses that expert's schedules everywhere, which is
what the guardrail means, and folding it into the tree would hide it.

---

## Untrusted input

The envelope carries a **taint** bit: an agent is tainted if its parent was,
or if it was started on untrusted input. It never clears — a new session id
does not wash it, because the bit lives on the turn rather than on the
session.

Taint flows down by inheritance and up by report. A parent that reads a
tainted child's report is reading attacker-influenceable text and is tainted
from that point. What confining a child buys is not a clean parent; it is
that the *action surface* of the agent that touched the untrusted input can
be made empty.

That is the case for pointing a narrow child at untrusted input: an agent
with `read_workspace_file` and nothing effectful has no tool an injection can
steer, and the worst it can produce is a misleading report — a bounded
artefact the parent reads as data. One agent reading the same page itself
gets steered mid-turn, before anyone sees anything.

**The guarantee belongs to the explicit read-only tool set, never to the
default.** A child that merely inherited the default narrowing still holds
`run_agent`, and `run_block`/`run_agent` reach outward-effecting blocks —
including arbitrary HTTP through `SendWebRequestBlock` — most of which carry
no sensitive-action flag. A descent-denied child is not an outward-safe
child, and nothing here should be read as saying it is.

Today nothing in the merged code *sets* the taint bit except chat-platform
sessions. The propagation exists ahead of the auto-mode approval gate — a
separate, unlanded change that decides when a turn needs a human's approval —
so that when it lands, a spawn cannot launder a tainted turn into a clean one.

---

## When a limit is hit

None of these produces an error page. Each returns a sentence to the calling
agent, which reports to the user in its own words.

| situation | what happens |
|---|---|
| Tree reaches its spend ceiling | The next spawn in that tree is refused: "This task has spent its budget; report what you have instead of starting more work." The turn already running continues and reports honestly. |
| Tree reaches 8 agents | Refused, saying explicitly that this is a lifetime budget for the request and waiting will not free a slot. |
| Fourth hop of depth | Refused: do as much as you can yourself instead of passing it on again. |
| A child crashes mid-turn | Existing behaviour — the stream errors and the turn is `failed`. Its recorded cost is charged to the tree; nothing was reserved, so nothing leaks. |
| Parent fans out past the per-user in-flight cap (15) | Existing behaviour — the spawn is rejected before any side effect and the tree's head-count is decremented. Children are rejected, never queued behind a parent waiting on them. |
| Two spawns race | Both hit the ledger; the second either fits or is refused. Admission is increment-then-check with rollback, so a race can over-admit by one — that is the bound's stated slack. |
| A session tries to resume or poll a child it did not create | Refused; the creator record does not match. |
| Target session already has a turn in flight | Refused as busy, rather than appending to the running turn's buffer. |
| Redis unreachable | Spawns fail closed; roots run. |
| A pre-existing child with no creator record | Cannot be resumed under the creator rule. A deliberate break: resuming it would mean resuming a child whose creator is unknown. |

---

## What changes for flows that work today

| flow | before | after |
|---|---|---|
| A delegated or handed-off expert posts to a chat platform, schedules a follow-up, calls an MCP tool, stores a skill or stores memory | inherited from the parent | **unchanged** — a delegate runs under its own identity, namespace and budget, so these stay in its default set |
| A delegated or handed-off expert connects an integration, sets up a webhook trigger, edits a preset, or deletes a folder, preset, schedule, skill or file | inherited | **refused** — binds the account or is irreversible; rare, visible, and recoverable, never silent data loss |
| A `run_sub_session` isolate posts outward, schedules, calls MCP, stores a skill, or writes memory | inherited | **refused** — this is the shared-namespace leak the design exists to close, and the change most likely to be noticed |
| A chain that used isolates to go past three hops | unbounded through isolates | **stops at depth 3** — isolates now count. A single isolate under a shallow chain is unaffected |
| Continuing a delegation to a teammate who is still working | queued into their running turn | **refused**, telling the caller to wait or start fresh |
| A sub-session's unanswered question | surfaced on Home | goes to its parent instead — sub-sessions now record their creator, and Home excludes delegated threads |

Whether the isolate memory-write default is too aggressive is a maintainer
call; loosening it is removing two entries from `ISOLATE_DENIED_TOOLS`.

---

## What this deliberately does not do

- No orchestrator, planner, router or manager agent, and no message bus, task
  board or work-item model beyond the queue that exists.
- No lateral edges — no sibling, peer or broadcast communication.
- No shared memory or team-wide context; memory scoping is untouched.
- No voting, debate or multi-round negotiation.
- No budget *reservation*: the ledger counts, so nothing has to be returned.
- No per-file workspace permissions. The workspace is user-scoped and flat,
  and the storage layer cannot back a per-file capability; what a child gets
  is a path to read, not a permission.
- No cross-user spawning of any kind.

---

## Limits worth knowing about

**`run_agent` starts a fresh tree.** `run_agent` is not on the denied list,
so a narrowed child keeps it, and it dispatches into the graph-executor
process where the envelope does not exist. If that graph contains an
`AutoPilotBlock`, the turn it starts is a *root*: unrestricted tools, depth
0, untainted, and a new tree with its own ceiling and head-count. A child's
narrowing, its place under the depth bound and its share of the tree's budget
are all reset. Nothing regresses against the previous behaviour, where
spawned turns had no envelope at all.
[#14265](https://github.com/Significant-Gravitas/AutoGPT/pull/14265) closes it by
carrying the tree through `ExecutionContext` into the executor process, the same
way expert identity already rides it for billing. Adding `run_agent` to the
denied list is the one-line alternative, at the cost of every delegate's ability
to run a graph.

**Graph spend is not on the ledger.** A graph run a child starts with
`run_agent` is charged in credits to the user's wallet and to the expert's
weekly budget, not to the tree. The ledger bounds LLM spend across the tree;
the wallet and the weekly budget bound graph spend; they are not yet one
number.

**Memory and the workspace are not tree-scoped.** Memory is shared along
identity — an isolate shares its parent's namespace, and every AutoPilot turn
for one user shares that user's. The workspace is user-wide. Neither can be
made a tree property without rebuilding the storage layer, so the design
closes the *write* side instead: isolates lose memory writes, and confining
workspace writes to the writing session's own folder is a named follow-up
below. A child can still read what its identity or its user can read.

**A tree bounds spend, not concurrency.** How many agents exist at once is
still governed by the per-user in-flight cap of 15. Across an organisation
that means total spend is bounded by the sum of users' own caps plus each
tree's overshoot, with nothing summing across trees — but "3000 agents" means
3000 experts across many users and schedules, not 3000 sessions running at
once under one root.

---

## Carried in the code but not reachable from any tool

So the shipped surface is not read as wider than it is:

- `tainted` propagates and is tested, but nothing reads it until the
  auto-mode gate lands.
- `SpawnRequest.max_seconds` has no caller, so `deadline_at` is always
  `None` and no deadline ever fires. There is no watcher either — the cancel
  function exists, nothing calls it on a timer.
- `SpawnRequest.tools` (the exact-set pin) has no caller, so the read-only
  quarantine shape described above is expressible in code but not selectable
  from any tool.
- `may_spawn=False` has no caller — all three spawn tools pass `True` — so no
  leaf is created in production.

---

## Designed, not built

The shape of what crosses an edge is settled and unimplemented. Today a spawn
carries free text plus the existing preamble; `SpawnRequest` carries only
`tools`, `may_spawn`, `shares_memory`, `max_seconds` and `born_tainted`. The
intended request is a **brief** with named slots — the task written for
someone who cannot see this thread, an inline payload for a child with no
tool to read a reference, workspace paths as references, what the child may
commit to, and what "done" looks like — and the intended response is a
**report** with a status, a summary, findings that carry a verbatim quote
each, the paths it wrote, and its taint bit.

Two slots earn a note. Saying what the child may commit to turns its boundary
decisions into closed questions instead of guesses: in an internal evaluation,
adding that list to an otherwise byte-identical prompt moved it from 9 correct
verdicts out of 18 to 15. Findings that carry a quote are what make a report
*checkable* rather than merely readable, which is what lets a parent audit a
child without re-doing its work.

Also named and not built: the deadline watcher; block filters in the envelope
(tools only today — `CopilotPermissions._parent` is a private attribute and
is dropped when the queue entry is serialised); workspace write confinement;
withholding the team roster from children that cannot spawn, which matters
because injecting the full roster is the one part of session creation that
grows with the number of experts.

---

## Where the code is

| what | where |
|---|---|
| Envelope, derivation, ledger, ceiling | `copilot/tree.py` |
| The one seam every turn passes | `copilot/executor/utils.py::dispatch_turn` |
| Tool refusal | `copilot/tools/base.py::BaseTool.execute` |
| Charging a turn to its tree | `copilot/token_tracking.py` |
| The running turn's envelope | `copilot/context.py::get_current_envelope` |
| Per-user daily/weekly caps and tier multipliers | `copilot/rate_limit.py` |
| Per-turn spend cap | `copilot/sdk/service.py::_resolve_dynamic_max_budget_usd` |
| The three spawn tools | `copilot/tools/{run_sub_session,delegate_to_expert,handoff_to_expert}.py` |
| Cross-expert chain check (loop detection) | `copilot/tools/expert_delegation.py::chain_refusal` |
| Tests | `copilot/tree_test.py`, `copilot/tools/sub_session_test.py` |
