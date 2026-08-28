# Agent collaboration architecture (TODO T15)

**Status:** plan → two parallel roasts (§11) → revised → slice built and
committed (§9, `5422a83e99`). Branch `pwuts/agent-collab-architecture`,
worktree `AutoGPT4`. Not pushed — push hold in effect, and §10 applies.

---

## 0. The answer, up front

T10 said *a team is a check*. That is true and it is the wrong unit. The unit
that scales is not the team and not the check; it is **the edge** — the thing
that is created when one agent causes another to exist or to act. Every hard
question in the brief (coupling, inheritance, position, scale) is a question
about what an edge carries, and every invariant is a statement that something
carried across an edge is monotone.

So the design is one primitive and four uses of it:

> **Every turn carries an Envelope; every tree has a Ledger.** The envelope
> is computed *only* from the spawning turn's envelope and the spawn
> request, by operations that can only narrow (intersection, minimum,
> increment-toward-a-cap, logical-or of a taint bit). The ledger is one
> metered counter per tree, checked at the single chokepoint every turn
> passes through. What crosses an edge is a brief down and a report up —
> and, because the architecture roast (§11.1) found three channels this
> claim missed, the design now *names* the channels it cannot close (memory
> along identity, a user-wide workspace) and closes them by denying
> children the write side rather than pretending they do not exist.

The four questions, in one line each:

| question | answer |
|---|---|
| **Context coupling** | Per edge, not global — but the *shape* is fixed by the platform and the *content* is chosen by the caller. What crosses is a **brief** (task, artefacts by reference, claimed authority, acceptance) and a **report** back. Transcript, memory and tool results never cross. |
| **Inheritance** | Split into three classes: **ceilings** (permissions, budget, deadline, depth, fan-out, dry-run, origin, taint, tenancy, billing route) inherit monotonically and can only narrow; **identity-bound** state (soul, memory namespace, voice, workflows, the expert's own weekly budget) is fixed by *which* expert the child is, never by the parent; **free within ceiling** (model, effort, wait time) is the caller's choice and is bounded by budget, not by tier. §3.2 is the table. |
| **Position** | A child has exactly two edges: in from its parent (brief) and out to its parent (report + progress). Nobody else can address it; it can address nothing but its parent and its own children. Siblings share data only through artefacts the parent hands both. `handoff` re-roots a node so the human becomes its parent. |
| **Scale** | The graph is a forest of trees rooted in *turns*. Every bound is a field on the turn or one Redis key per tree, checked at `schedule_turn` in O(1). Budget is a **tree ledger** (metered accounting checked at turn start — not a conservation law the executor cannot enforce). The one non-local thing today — the roster injected into every prompt — is withheld from leaves and scoped for everyone else. No agent ever sees the whole graph. |

And the thing I am not going to pretend: at N = 3000 the *concurrency* limits
are per user (in-flight 15 on every spawn path — `schedule_turn` uses the
in-flight cap, not the running cap of 5) and they are the actual bound on how
many nodes exist at once. That is correct — a tenant's cap is local to the
tenant — and it means "3000 agents" is 3000 *experts* across many users and
schedules, not 3000 concurrently running sessions under one root. A design
that promised the latter would be lying about the executor.

---

## 1. What exists today — real, rendered-only, or absent

All paths relative to `autogpt_platform/backend/backend/`. Read, not assumed.

### 1.1 Real and enforced

| thing | where | what it actually does |
|---|---|---|
| Expert identity | `copilot/expert_context.py::build_expert_identity_suffix` | Soul appended to the system prompt every turn; fails closed if the expert is missing/archived; tenancy checked. |
| Roster | `expert_context.py::_team_context` | `<team_context>` with *every* non-archived expert, injected into the first user message. O(N) tokens per session. |
| `delegate_to_expert` | `copilot/tools/delegate_to_expert.py` | New `ChatSession` bound to the target, one turn via `run_copilot_turn_via_queue`, blocks ≤ 300 s (`MAX_TOOL_WAIT_SECONDS`), pollable through `get_sub_session_result`. |
| `handoff_to_expert` | `copilot/tools/handoff_to_expert.py` | Same machinery, `status="transferred"`, caller cannot poll; the child's questions reach Home (`copilot/db.py:951` re-admits `handed_off_from_expert_id IS NOT NULL`). |
| `run_sub_session` | `copilot/tools/run_sub_session.py` | Same-identity isolate; hardcodes `origin="automation"`; does **not** write `delegated_by_session_id`. |
| Chain bound | `copilot/tools/expert_delegation.py::chain_refusal` | `MAX_DELEGATION_DEPTH = 3`, walks `delegated_by_session_id` upward with a `seen` set; refuses handing back to an expert already in the chain. |
| Poll capability | `get_sub_session_result.py::_in_caller_scope` | A cross-scope sub is readable only by the session named in its `delegated_by_session_id`. This *is* an edge capability — the design keeps it. |
| Provenance | `copilot/model.py::ChatSessionMetadata` | `delegated_by_expert_id`, `delegated_by_session_id`, `handed_off_from_expert_id`, `origin`, `dry_run`, `llm_auth_provider`, `llm_credential_id`, `pending_question`. |
| Tool groups | `copilot/tools/__init__.py::expert_tool_disabled_groups` | `expert_admin` hidden *and refused* for expert sessions; `experts` for plain sessions; `delegation` behind `HIRE_EXPERTS`. Enforced at `execute_tool`. |
| Staffing guard | `tools/expert_proposal.py::autopilot_session_guard` | Hire/raise/update require `origin == "interactive"` and no `expert_id`. |
| Permissions model | `copilot/permissions.py::CopilotPermissions` | `merged_with_parent` intersects tools and chains block filters. **Only `AutoPilotBlock` calls it** (`blocks/autopilot.py:808`), through a process-local contextvar. |
| Concurrency | `copilot/active_turns.py`, `executor/utils.py::schedule_turn` | Per-user running cap (5) and in-flight cap (15). A parent blocked on a child holds a running slot. |
| Per-user spend cap | `copilot/rate_limit.py::get_remaining_usd_budget`, `sdk/service.py::_resolve_dynamic_max_budget_usd` | Daily/weekly microdollar caps per user; per-query SDK `max_budget_usd = min(static cap, remaining)`, floored so the "wrap up" reminder can fire. |
| Expert weekly budget | `api/features/experts/scheduling.py::enforce_expert_run_budget` | Credits, ISO-week Redis counter, **gates expert-attributed graph runs only** (schedules/triggers). Breach pauses schedules, never chat. |
| Memory namespace | `copilot/graphiti/client.py::derive_memory_group_id` | Keyed on the *session's* `expert_id`. No cross-namespace read path. |
| Tenancy | `schema.prisma` `Organization`/`Team`/`TeamMember`; `experts_db.resolve_private_expert_tenancy` | Org/team models exist; experts are `PRIVATE`-only today and TEAM/ORG visibility fails closed. |

### 1.2 Rendered but never read

- **`EXTERNAL_ACTION_APPROVAL_RULE`** (`api/features/experts/models.py:21`).
  Grep across the backend: rendered by `expert_context.py:100`, returned by
  `experts_db.py:136`, read by **nothing**. T10's finding stands (TODO T16).
  This plan does not lean on it anywhere.
- **`toolProfile Json?`** on `Expert` — a column with no reader in the copilot
  path. Noted because it is the natural home for a per-expert permission
  ceiling and is currently dead.

### 1.3 Absent

- **Permission narrowing at delegation.** All three spawn tools pass
  `permissions=get_current_permissions()` — the parent's own filter, verbatim
  — and offer no way to give a child *less*. Precisely: a child's authority
  *equals* its parent's; it does not widen, and it cannot narrow. That is why
  intersection is the envelope's default rather than an option. Also:
  `CopilotPermissions._parent` is a `PrivateAttr` and is dropped when the
  entry is serialised onto the queue, so `AutoPilotBlock`'s block-filter
  chain does not survive the hop (the tool whitelist does — it is stored as
  an intersected list). *(Logged in the shared findings ledger as
  security-sensitive; see §11.)*
- **Budget for delegated LLM turns.** The delegate tool's own docstring says
  it: "only graph executions accrue weekly spend; the delegated
  conversation's own LLM cost does not." A child's turn draws from the
  *user's* daily cap with no relation to what the parent was allotted.
- **Deadline.** None inherited. A child may run to `MAX_TURN_LIFETIME_SECONDS`
  (6 h) regardless of the parent's wait.
- **Fan-out bound.** A node may spawn any number of children per turn; only
  the per-user in-flight cap stops it.
- **Depth that survives `run_sub_session`.** The chain bound is computed by
  walking `delegated_by_session_id`, and a same-identity sub-session does
  not write that field. Depth 3 is therefore a bound on *consecutive
  cross-expert* hops, not on the tree — which is why the envelope carries
  depth as a field every spawn kind increments. *(Logged in the shared
  findings ledger as security-sensitive; see §11.)*
- **Taint.** There is no taint tracking on `dev`. T9's (`AutoGPT3`,
  `pwuts/autopilot-auto-mode`) is branch-only. `child_session_origin` returns
  the parent's origin — so an *interactive* parent mints an *interactive*
  child — which is the laundering path T9 found.
- **Pods do anything.** `ExpertPod` is a folder on one page.

---

## 2. The model

### 2.1 Nodes and roots

A **node** is a `ChatSession` running under one identity (an expert, or
AutoPilot when `expert_id is None`). That is unchanged.

A **root** is a node whose turn was started by something that is not another
node: a human typing, a schedule firing, an `AutoPilotBlock` in a graph, a
chat-platform message. Everything else is a **child**, created by exactly one
of the spawn tools from exactly one parent turn.

The accounting unit is the **root turn**, not the session. A long-lived
expert thread is a root many times over; each root turn opens a fresh tree
with a fresh ceiling. This is what makes the whole thing scale-invariant: no
tree outlives the turn that rooted it, and no bound has to be reconciled
across turns. Two consequences the roast forced me to state: a *resumed*
child (re-delegating into a prior `delegated_session_id`) is a node of the
**caller's current** tree, not of the tree that created its session — the
tree id is a property of the turn, never of the session row; and a
`handoff` turn is the last node of the parent's tree, after which every
human turn typed into the handed-off thread is a new root. That is what "re-
rooting" means mechanically, and it is why the handed-off thread's `origin`
does not have to lie about who drives it.

### 2.2 Edge kinds

Four, and only four. Three already exist; the fourth is T10's and slots in
as a degenerate case.

| kind | tool | identity of child | what returns | may spawn | who the child escalates to |
|---|---|---|---|---|---|
| **delegate** | `delegate_to_expert` | the target expert | report, pollable by the parent | yes, within envelope | its parent |
| **isolate** | `run_sub_session` | same as parent | report, pollable by the parent | yes, within envelope | its parent |
| **handoff** | `handoff_to_expert` | the target expert | nothing (`transferred`) | yes, within envelope | **the human** — the node is re-rooted |
| **consult** | T10 `consult_teammate` | a fixed frame + the target's declared boundaries | a report with `findings` (= T10's verdict), inline | **no** — `tools = ∅`, no session | n/a (stateless) |

`handoff` is the only edge that changes who a node reports to, and it does so
in exactly one direction: toward the human. There is no edge that makes a
node report to a sibling, a peer, or a third expert. All four carry the
same brief and return the same report shape — §2.6 is the table of what
each one requires and promises.

### 2.3 The Envelope, the Ledger, and the Provenance — three carriers, three lifetimes

The first draft put one object on the session row and asked it to be both
immutable and CAS-mutated. The roast (§11.1 findings 10, 15, 16) showed that
cannot work: session metadata is documented immutable, is served from a
12-hour Redis cache, and is read by the spawn tools as an in-memory object
for the whole turn — and "per-turn fan-out" is not representable on a row
that outlives the turn. So the state is split by lifetime:

**(a) The Envelope — per turn, immutable, on `CoPilotExecutionEntry`.**

```python
class TurnEnvelope(BaseModel):
    tree_id: str                  # the root turn's id; every node in the tree shares it
    depth: int                    # 0 for a root
    tainted: bool                 # spawner.tainted OR born-tainted
    tools: frozenset[str] | None  # None = unrestricted root; else ⊆ spawner's
    block_filters: list[BlockFilter]  # appended, all must pass (serialisable, unlike _parent)
    deadline_at: datetime | None  # min(spawner.deadline_at, now + requested)
```

It is derived at exactly one place — `schedule_turn` — from the **spawning
turn's** envelope (held in the executor's contextvar for the running turn,
next to `permissions`) and the spawn request. Never from a session row,
never from anything the model supplies. A root turn (HTTP chat, scheduler,
`AutoPilotBlock`, chat-platform bot) gets a fresh envelope with a fresh
`tree_id`. The derivation:

```
derive_child_envelope(spawner: TurnEnvelope, req: SpawnRequest) -> TurnEnvelope | Refusal
  tree_id       = spawner.tree_id
  depth         = spawner.depth + 1                       ; refuse if > MAX_DEPTH
  tainted       = spawner.tainted or req.born_tainted
  tools         = (spawner.tools or ALL) ∩ (req.tools or ALL − DESCENT_DENIED)   ; §3.6
  block_filters = spawner.block_filters + req.block_filters
  deadline_at   = min(spawner.deadline_at, now + req.max_seconds)               ; refuse if past
  dry_run       = spawner.dry_run or req.dry_run          ; on the session, as today
  billing route, tenancy, user_id = spawner's              ; refuse a request to change them
```

Every operation is monotone. There is no field a request can raise, and
there is no code path that mints a child turn without passing here — because
`schedule_turn` is the one function every spawn, resume, handoff and
`AutoPilotBlock` dispatch already calls (`executor/utils.py:306`).

**(b) The Ledger — per tree, mutable, one Redis key.**

```
tree:<tree_id>  →  { ceiling_microdollars, spent_microdollars, nodes, max_nodes, expires_at }
```

Opened by the **first spawn**, not by the root: `ceiling = min(configured
tree ceiling, user's remaining daily/weekly cap)`; `max_nodes` from config;
`nodes = 1` for the root; TTL = the maximum turn lifetime. Roots never
touch the ledger — the HTTP route, the scheduler and `AutoPilotBlock` pay
nothing for this change, and a tree that never spawns never exists. Two
operations, both single Redis hash commands:

- `admit(tree_id)` at `dispatch_turn` for depth > 0: refuse unless
  `spent < ceiling`; `HINCRBY nodes` and roll back if it passed
  `max_nodes`. Increment-then-check with rollback is the same non-locked
  admission `acquire_turn_slot` uses; the over-admit-by-one under a race is
  the bound's stated slack. This is the fan-out bound (per tree, which *is*
  representable) and the budget bound.
- `charge(tree_id, microdollars)` from `token_tracking` when any turn in the
  tree records its cost (`token_tracking.py:228` already has the session and
  the cost in hand). A root's own cost lands only if its tree exists — i.e.
  if it spawned — which is the only case the ceiling is for.

This is an **accounting** invariant, not a conservation law: a node that
would exceed the ceiling does not get a turn. Overshoot is bounded by one
turn's spend, which is bounded by the SDK per-query cap (and by
`agent_max_turns` on Codex, where there is no USD cap at all — §8.2). It is
transport-independent, needs no CAS on any row, no floor arithmetic, and no
lease return: nothing was reserved, so nothing has to come back, and a
parent can never be starved by its own children's *reservations* — only by
their actual spend, which is the thing we want to bound.

Redis unavailable → `admit` fails closed for depth > 0 (a spawn is refused;
the human's own root turn still runs, exactly as the per-user rate limit
behaves today).

**(c) The Provenance — per session, immutable, already exists.**

`delegated_by_session_id` is written by **all three** spawn tools (today
`run_sub_session` omits it — that omission is the unguarded lateral edge in
§11.1 finding 4). It is the *creator capability*: resuming, polling or
cancelling a child requires the caller's session id to match it. A session
cannot resume itself (its provenance names its parent, not itself) and
siblings cannot resume each other, which also removes the self- and
mutual-deadlock the roast found (finding 11).

`origin` stays what it is today. The first draft made every child
`automation`; that would have stamped handed-off threads the human types in
as machine-driven forever (finding 15) *and* made every delegated thread
resumable by any same-expert session (finding 4). Taint is its own bit on
the turn envelope; origin no longer has to carry it.

**Spawns never queue.** `run_copilot_turn_via_queue` has a path that, when
the target session already has a turn in flight, appends the message to that
turn's pending buffer — executing the child's prompt inside *another* turn
under *that turn's* permissions (finding 3). For a spawn or resume that path
is refused (`target_busy`) rather than taken. A fresh child session can never
be busy; a resumed one can, and then the caller is told so.

### 2.4 The Brief — what a parent chooses to send

This is the context-coupling knob, and it is per edge. It is *not* free text
with a system-context prefix (what exists today); it has slots, and the slots
are the T10 lesson turned into a schema:

```
SpawnRequest:
  task: str                     # what to do, written for someone who cannot see this thread
  content: str | None           # inline payload, bounded — for edges whose child has no tool to read a reference
  artefacts: list[str]          # workspace paths — REFERENCES, not capabilities (see below)
  authority: str | None         # what the caller asserts the child may assume/commit to  (T10) — optional
  acceptance: str               # what "done" looks like, in a form the child can check itself against
  may_spawn: bool = False       # default: the child is a leaf (and gets no roster — §3.4)
  max_seconds: int              # deadline request — clamped, never raised
  tools: list[str] | None       # exact set (a quarantine preset) — intersected, never widened
  grant: list[str]              # descent-denied tools to hand down — intersected with what the spawner holds
  shares_memory: bool           # the child writes the spawner's namespace (an isolate) — memory writes withheld
```

`content` exists because §2.6's schema test found it missing: a tool-less
edge (T10's consult) cannot follow a reference, so the payload has to ride
in the brief. It is bounded (T10 uses 8,000 chars) and it is the *only*
slot whose size is bounded by the platform rather than the caller.

On `artefacts`, corrected after the roast (§11.1 finding 2): the workspace is
**user-scoped and flat** (`util/workspace.py`) — any session can read, and
write, any `/sessions/<id>/...` path of the same user. There is no per-file
capability to grant, and the first draft's "READ capability on exactly
these" was wrong. What the design does instead: artefacts are *references*
(data by path, not by paste), and **children may only write inside their
own session folder** — a one-line check in the workspace write path when
the turn's depth > 0. That closes the upward write channel (a child
planting `/sessions/<parent>/notes.md` for the parent to read as its own)
without inventing a capability system the storage layer cannot back.

On `authority`, corrected after the value roast (§11.2 finding 8): it is
optional. A child summarising three files has no commitment surface, and
T5's lesson is that a rule you get for free is noise. The slot is there for
the tasks that have one.

Why `authority` is a slot and not prose: T10's roast killed the original
reviewer design because the veto question ("does this commit the company to
something the user did not authorise?") was **unanswerable from what
isolation let the reviewer see**. The fix was to carry the *claimed
authority* across the boundary so the check became a closed question. That
generalises: every child is in that reviewer's position for its own task. A
brief that says "you may quote up to $500; you may not promise a date" makes
the child's boundary decisions closed questions. A brief without it makes the
child guess, and a guessing child is a child that either over-commits or
stalls and asks. The slot costs the caller one sentence and is the single
highest-leverage thing that crosses the edge.

That was argued from first principles when this file was first written; T10
then measured it the same night. On a grid of 6 fixed drafts × 4 arms × 3
runs, run twice with all 24 cells identical: a naive "is anything wrong with
this draft?" prompt with **no** authority supplied scored 9/18 (6 false
blocks, 3 misses); the byte-identical prompt **plus an authority list**
scored 15/18 (3 false blocks, 0 misses). The cleanest case is a
covered/uncovered pair — same draft text, opposite correct verdicts — that
no receiver of any quality can separate without the field, and every arm
that had it separated 3/3. Supplying claimed authority is worth 6/18 on an
otherwise identical boundary crossing. That is the evidence for making it a
slot rather than hoping the caller mentions it.

Why `acceptance` is a slot: T5 showed that what a fixed structural audit
reliably measures is **process discipline, not correctness**. An acceptance
line is the child's own process check — "did I produce the thing that was
asked for in the form that was asked for" — which is exactly what a report can
be audited against without a persona and without a rubric. It is deliberately
*not* a correctness oracle, and the design does not pretend a parent can
verify a child's answer is *right* from the report alone.

What deliberately does **not** cross: the parent's transcript (noise, cost,
and taint in one package), the parent's memory namespace (identity-bound),
the parent's tool results (they are in the transcript), and any pointer that
would let the child read the parent's session. Coupling is *explicit,
enumerated, and one-directional*.

### 2.5 The Report — what comes back

```
Report:
  status: done | needs_input | needs_approval | failed | out_of_budget | out_of_time
  summary: str
  findings: list[{quote: str, reason: str}]   # verbatim evidence — the bait-audit shape (T5: 97.9% self-agreement)
  artefacts: list[str]            # what the child wrote, by path (list_sub_workspace_files enumerates; it does not scope)
  tainted: bool                   # the child's envelope bit — a reader of this report inherits it
  asked: str | None               # the question, when status is needs_input / needs_approval
```

`findings` is the field that makes a report *checkable* rather than
*readable*: each entry carries a verbatim quote the parent can locate in
the artefact. It is T5's bait-audit shape generalised — the form that
scored 97.9% self-agreement where holistic rubrics scored ρ = 0.24 — and it
is what T10's `block` verdict is made of. §2.6 shows the mapping.

`tainted` on the report is the half of the taint rule the first draft
missed: taint flows **down by inheritance and up by report**. A parent that
reads a tainted child's report is reading attacker-influenceable content
and its own turn is tainted from that point. What the tree confines is the
*action surface* of the node that touched the untrusted input (§3.5), never
the taint status of the nodes above it.

`needs_input` and `needs_approval` are **the escalation channel, and they go
up the tree.** When a `needs_approval` reaches a root and becomes a human
approval card, the card's headline and arguments must be composed
server-side from the envelope and the tool registry — never from a
model-supplied field. T9's roast recorded (findings ledger) that the existing
review card can render one string while a different payload executes; an
escalation surface that inherits that would make the tree's one human
checkpoint the weakest node in it. A child cannot ask a human — the delegate preamble already
says so, and `copilot/db.py:951` already hides a delegated child's
`pending_question` from Home. What is new is that the parent treats a
`needs_*` report as *its* next step (the delegation supplement already says
"delegated work is yours to land"), and at the root — the only node with a
human — it becomes an `ask_question` / a gate card. Escalation therefore
travels the same edges as reports, in the same direction, and there is no
node from which it can go sideways.

### 2.6 Edge types — the brief/report is the mechanism; `consult_teammate` is edge type #1

Reinier read T10 and this file side by side and saw a large, vaguely
defined overlap. It is exact, not vague: T10's `consult_teammate` passes
work plus the authority it claims and receives a structured verdict with
verbatim quotes. That *is* a brief and a report. T10 built one concrete
instance of the general mechanism before the general mechanism was
written down, and this section subsumes it rather than running two payload
shapes side by side.

**The schema test.** Can the brief/report express T10's consult exactly,
without special-casing? Slot by slot:

| T10 consult | brief / report | fit |
|---|---|---|
| `content` — the draft, inline, ≤ 8,000 chars | `SpawnRequest.content` | **was missing**; added in §2.4. A tool-less child cannot follow an artefact reference, so the payload must ride in the brief. |
| `question` — "does this commit the company to anything the user did not authorise?" | `task` | direct |
| the caller's declared authority | `authority` | direct — and required for this edge type (optional in general) |
| implied: "flag any commitment not covered" | `acceptance` | direct |
| verdict `pass` | `status = done`, `findings = []` | derived, not stored |
| verdict `block` + `quotes` | `status = done`, `findings = [{quote, reason}, …]` | **needed `findings`**; added in §2.5 |
| verdict `insufficient` naming the missing fact | `status = needs_input`, `asked = <the missing fact>` | direct — and it is the same escalation channel every edge uses |

Two slots were missing and both were the schema's fault, not the edge's.
Both additions turned out to be general: `content` is what any tool-less
edge needs, and `findings` is what the refund-audit leaves in §3.5 return
— the general report was already going to need a quote-carrying channel
and had not admitted it.

**T10's constraints, taken as constraints on the general design.** T10
removed three things deliberately, and each removal was a reason:

- *Tool-less by construction, so recursion is impossible.* In envelope
  terms: `tools = ∅` (the empty set is the strongest ceiling; `None` is the
  weakest) and no session. A consult is a node on the ledger — it is
  admitted and its one completion is charged — but it has no turn that
  could spawn. General edges *do* have tools, and the honest statement is
  that for them recursion is **bounded, not impossible**: depth ≤ 3, nodes
  ≤ `max_nodes`, spend ≤ ceiling, all checked at the chokepoint. That is
  weaker than T10's guarantee, and it is acceptable only because work
  needs tools and a check does not. An edge type declares which of the two
  it is; nothing is allowed to be "a check with tools".
- *No memory read, no memory write.* Reads would need a tool loop; writes
  would leak one identity's context into another's store one consult at a
  time. In the general design memory *writes* are descent-denied by
  default for every child (§3.6, and §11.1 finding 1 is why); memory
  *reads* are allowed to work edges because a delegated expert without its
  own memory is not that expert. A check edge gets neither.
- *A fixed, soul-free judge.* T5 showed persona judges swing the measured
  dimension across the full range, and T10 then measured the teammate's
  identity contributing **3/18** over "a check happens at all" — what the
  identity actually supplied was its *declared boundaries*, i.e. policy,
  not judgement. So the general rule: **identity on an edge buys
  structural things — a memory namespace, integrations, workflows — and
  declared policy. It is never assumed to buy judgement.** Check-type edges
  run a fixed frame with the target's boundaries as policy input; work-type
  edges run under the target's soul because the soul is where the
  structural things are attached.

**What T10 measured, used here instead of asserted.** Supplying the
authority list: +6/18 on a byte-identical prompt (§2.4). The structured
frame vs a naive "is anything wrong?": no score gain (12 vs 9, both 15 with
policy) but a scope gain — 0–1 off-topic objections in 12–15 against 3 in
12 (§3.5). Identity over "a check at all": 3/18. Together these say the
value of the mechanism is in *what crosses the edge* (authority) and *what
shape comes back* (in-scope findings), not in who is on the other side.
That is the whole reason the brief and report are typed and the edge types
are few.

**How the two compose, and what a second edge type looks like.** An edge
type is a tuple: an envelope preset, the brief slots it requires, the
report shape it promises, and whether its frame is fixed or identity-borne.

| edge type | envelope preset | required brief slots | report | frame |
|---|---|---|---|---|
| **#1 consult** (T10, shipped) | `tools = ∅`, leaf, no session, one completion | `content`, `authority`, `task` | `done` + `findings` / `needs_input` + `asked` | fixed; target's boundaries as policy |
| **#2 quarantine read** (the §3.5 leaf) | `tools = {read_workspace_file}`, leaf, born-tainted | `artefacts`, `acceptance` | `done` + `findings` with quotes; `tainted = true` | fixed |
| **#3 delegate / isolate** (exists) | `ALL − DESCENT_DENIED`, `may_spawn` optional | `task`, `acceptance` | any status; `artefacts` | identity-borne (delegate) / spawner's (isolate) |
| **#4 handoff** (exists) | as #3, re-roots | `task` | none to the spawner | identity-borne |

Edge type #2 is the one this design adds and the one §3.5 needs: it is
#1 with one read tool and an artefact list instead of inline content. It
is what turns "read 200 untrusted threads" into 14 leaves that cannot
act. Anyone proposing a fifth payload shape in a month should be able to
write it as a row in this table; if they cannot, it is not an edge type,
it is a framework.

---

## 3. The four questions

### 3.1 Context coupling — where is the useful middle?

It is not a scalar. The evidence:

- Too strong: giving the child the parent's transcript. Costs tokens
  linearly in transcript length per child, carries every injected
  instruction the parent ever read, and — per T5 — more text does not buy
  better behaviour (the 7.6k kernel beat the 19.6k full soul).
- Too weak: T10's reviewer, which could not answer its one question.

The middle is **typed, explicit, caller-authored, and per edge** (§2.4). The
platform fixes what *kinds* of things may cross (task, artefact refs,
authority, acceptance) and forbids the rest. The caller decides how much of
each to put in. That is the same shape as a good delegation between people:
you do not forward your inbox, you write a brief.

Is it one setting or a property of each relationship? **A property of each
edge**, with a fixed schema. There is no team-wide "coupling level" because
two edges from the same parent legitimately need different briefs: the
research child needs the source docs by reference and no authority; the
"draft the reply" child needs the authority line and no docs.

Artefacts are the one place siblings can share anything, and they do it
through the parent: the parent puts a workspace file id in both briefs. Data
flows; instructions do not. That is a deliberate asymmetry — data is inert
until a model reads it, and the model that reads it is one whose envelope
already carries the taint bit of the tree it is in.

### 3.2 Inheritance — the table

| parameter | class | rule on descent | must NOT change | enforced where |
|---|---|---|---|---|
| `user_id`, `organization_id`, `team_id` | ceiling | identical | ✔ | `create_chat_session` copies from parent; child cannot pass its own (exists) |
| billing route (`llm_auth_provider`, `llm_credential_id`) | ceiling | identical | ✔ | exists: resume checks refuse a mismatch; spawn copies |
| `dry_run` | ceiling | `spawner or requested` | ✔ (true is sticky) | exists for delegate/isolate/handoff |
| `origin` | structural | copied as today (`child_session_origin`) | never a taint carrier again | unchanged — taint has its own field |
| taint | ceiling | `spawner.tainted or born_tainted`; **up** via report | ✔ (never clears) | **new**: turn envelope; T9's gate reads it as a source; chat-platform sessions born tainted |
| tools | ceiling | `(spawner or ALL) ∩ (requested or ALL − DESCENT_DENIED)` | ✔ | **new**: derived at `schedule_turn`; **enforced in `BaseTool.execute`** so both engines refuse, not just hide (§11.1 finding 5) |
| blocks | ceiling | spawner's filters + requested, all must pass | ✔ | **new**: serialisable `block_filters` list replaces the `_parent` PrivateAttr the queue drops (finding 6) |
| budget | ceiling | one metered counter per tree; a turn is admitted iff `spent < ceiling` | ✔ (Σ tree ≤ ceiling + one turn's overshoot) | **new**: Redis tree ledger, `admit` at `schedule_turn`, `charge` from `token_tracking` |
| fan-out | ceiling | one counter per tree; `nodes < max_nodes` at admit | ✔ | **new**: same ledger key — per-tree, not per-node-per-turn, because only the former is representable |
| depth | ceiling | `spawner + 1 ≤ MAX_DEPTH` | ✔ | **new**: turn envelope; applies to isolate and resume, which today it does not |
| deadline | ceiling | `min(spawner.deadline, now + requested)` | ✔ | **new field + new watcher** — `enqueue_cancel_task` exists, nothing calls it on a timer today (finding 14) |
| escalation target | structural | parent; the human only at a root; handoff re-roots | ✔ | exists (Home predicate `copilot/db.py:951`) + report status |
| creator capability | structural | `delegated_by_session_id` = spawner | only the creator may resume / poll / cancel | **change**: `run_sub_session` starts writing it; resume paths require it |
| identity (soul, name, voice) | identity-bound | the *target expert's*, from its row | caller cannot inject into the system prompt | exists: suffix built from the expert row; `safe_caller_name`; `fence_voice_preferences` |
| memory namespace | identity-bound | the target expert's — **shared along identity, not isolated by the tree** | a child sharing its spawner's namespace does not get the write side | exists: `derive_memory_group_id(session.expert_id)`; **new**: `memory_store` / `add_understanding` withheld from isolates; `store_skill` descent-denied (finding 1) |
| workspace | user-bound | reads are user-wide (data by reference) | children write only inside their own session folder | **new**: write confinement at depth > 0 (finding 2) |
| workflows, integrations | identity-bound | the target expert's | — | exists |
| expert weekly budget | identity-bound, orthogonal cap | the target expert's own counter; a breach pauses that expert for every tree | — | exists: `enforce_expert_run_budget` on graph runs; deliberately non-local (§11.1 finding 13, rejected) |
| model, thinking effort | free within ceiling | caller's choice, routed by `resolve_model_route` | — | budget is the bound, not tier |
| wait time (`wait_for_result`) | free within ceiling | caller's choice ≤ `MAX_TOOL_WAIT_SECONDS` (300 s) | — | exists |

Argued explicitly, since the brief asked: **model tier is not a ceiling.** A
child that needs a stronger model to do its narrow job should have it; what
must not widen is the *spend*, and the tree ledger bounds that regardless of tier.
Making tier monotone would push callers to over-provision the root so
children can be strong — which raises spend, the thing we are trying to
bound. The subscription-tier constraint on *which* models exist is tenancy
and is already identical down the tree.

Second argument: **the expert's own weekly budget stays orthogonal** rather
than being folded into the tree ledger. It is a per-identity churn guardrail
("this expert may not burn more than X/week on schedules"); the ledger is a
per-tree spend bound. Two different questions. Today a graph run started by
a delegated turn already counts against the target expert's week and can
pause her schedules for every tree (§11.1 finding 13) — that is the
guardrail doing its job, and folding it into the tree would hide it.

### 3.3 Position in the communication graph

A spawned node lands at exactly one place: **as a leaf under the turn that
spawned it**, with:

- one inbound edge: the brief, delivered once as the first user-role message,
  framed by the existing preamble that says a colleague — not the user — sent
  it;
- one outbound edge: the report, plus the existing poll capability
  (`delegated_by_session_id`) that lets the parent — and only the parent —
  wait on it;
- the right to spawn its own children, iff `children_remaining > 0`.

Who can address it: its parent (poll/cancel), and the human, who owns every
session. Not siblings, not peers, not other experts on the roster, not a
"manager". There is no lookup by which any other node discovers it exists.

Why no sibling or peer edges — corrected after the value roast (§11.2
finding 5), which rightly pointed out that intersection *is* a monotone
answer to "which envelope applies": lateral edges are possible, and they
are rejected for two reasons that are not impossibility. First, a graph
with lateral edges is not a tree, and two nodes that can each wait on the
other can deadlock — the architecture roast found exactly that shape
already latent in the unguarded `run_sub_session` resume path (§11.1
finding 11), and the tree rule is what removes it. Second, every lateral
edge is a second enforcement surface for a hop the parent already carries
at the cost of one extra report. "We chose not to, because it doubles the
surface and reintroduces cycles" is the honest sentence.

`handoff` is the deliberate exception: it re-roots. The handoff turn itself
is the last node of the parent's tree and is admitted against the parent's
ledger. Every later turn in the handed-off thread — the human typing, a
follow-up — is a new root with a new tree and a new ceiling from the user's
cap. Ownership moves; authority does not widen; nothing is "converted".

What the roster (`<team_context>`) is for under this model: **addressing,
not context.** It tells a node which identities it may spawn a child *as*. It
is not a channel and it does not make those experts reachable — a listed
expert has no session until the caller spawns one.

### 3.4 Scale invariance — 3 agents and 3000

**Three.** A founder with AutoPilot, an Ops expert and a Growth expert.

```
human ── root turn (AutoPilot, interactive)   tree ledger{ceiling $1.00 (= default daily cap), nodes 1/8}
           ├── delegate → Ops    envelope{depth 1, leaf, deadline +5m, tools = ALL − DESCENT_DENIED}
           │                     brief{task, authority: "no refunds, no dates", acceptance: "a reply draft"}
           │                     report{done, artefact: /sessions/<ops>/draft.md, tainted: false}
           └── delegate → Growth envelope{depth 1, may_spawn, deadline +10m}
                                 └── isolate (Growth) envelope{depth 2, leaf}
```

Every tool the user sees is the tool that exists today. What changed is that
Ops *cannot* delegate onward (leaf), *cannot* schedule anything or post
outward (descent-denied), *cannot* start a turn once the tree has spent its
$1.00, *cannot* run past the parent's deadline, and *knows* it may not
promise a refund because the brief said so. The ToolChain card shows the
envelope next to the delegate row. Total spend ≤ $1.00 plus at most one
turn's overshoot. Note the numbers: the default daily cap is $1.00
(`config.py:361`), which is why the first draft's "$2 lease with a $0.60 and
an $0.80 child" could not exist (§11.1 finding 9) — a ledger that meters
actual spend has no such problem; four small turns fit where two reserved
leases could not.

**Three thousand.** An organisation of 400 users, 3000 hired experts, a few
hundred schedules firing across the day, plus graphs with `AutoPilotBlock`s.

- There is no global object. There is a forest; each tree is rooted in a
  turn; each tree is bounded by `MAX_DEPTH = 3` and `MAX_TREE_NODES` (one
  counter on the tree's ledger key). Children are leaves by default; a
  parent that wants a deeper tree says `may_spawn` and the depth bound
  still applies.
- Spawn cost is O(1): one Lua call on one Redis key plus the session
  create that exists today. The O(depth) provenance walk goes away because
  depth is a field. **But** — corrected after §11.1 finding 12 — the session
  create is O(N) today because every new session's first message injects
  the full roster via `list_experts` with workflows joined, and
  `resolve_target_expert` / `unknown_target_message` each re-list on a
  miss. So the roster change below is not deferrable scale hygiene; it is
  on the hot path of every spawn.
- Spend: every root ceiling is clamped to the user's remaining daily/weekly
  cap (exists); every turn in the tree is admitted against the tree's
  metered counter; therefore `Σ spend over the org ≤ Σ over users of their
  caps + one turn's overshoot per tree`, with no component ever summing
  across trees.
- Concurrency: per-user in-flight cap of 15 on every spawn path (exists).
  3000 experts is 3000 *rows*; the number of live sessions is bounded by
  users × cap.
- Addressing: the roster injection is the one O(N) thing and it breaks
  first (3000 experts is ~150k tokens of roster in *every* first message).
  Two changes: **leaves get no roster at all** — a child that cannot spawn
  has no use for one, and leaves are the default, so the common child
  session drops the O(N) read entirely; and for nodes that may spawn,
  `<team_context>` lists at most the caller's **pod** (or the first K by
  recent use when unpodded), with `list_team` gaining a query so an expert
  is *found*, not enumerated. `resolve_target_expert` already accepts
  name-or-id. This gives pods a real job — **an addressing scope** — without
  making them an org chart: a pod is who you can spawn without looking
  anyone up.
- Escalation: each root is one Home item; a tree of depth 3 under a
  scheduled expert produces at most one "needs you" per root turn, because
  `needs_*` propagates up and lands once.
- Tenancy: org/team models exist; experts are PRIVATE-only today. Making an
  expert TEAM-visible is a tenancy change (who may *spawn as* it), not a
  collaboration change, and nothing in this design depends on it — but the
  envelope's `user_id` rule has to become "the spawning user is a member of
  the expert's team", which is a one-line change in `resolve_target_expert`.

What is *not* scale-invariant and I am saying so: the overshoot. A tree's
ceiling is checked at turn start, and a turn that is admitted may spend up
to the SDK per-query cap (floored at $0.50, times up to three stream
attempts) before the ledger sees the charge. At the default $1.00 daily cap
that overshoot is the same order as the ceiling. The ledger makes the
*bound* honest; it does not make the *granularity* fine, and on Codex —
where there is no USD cap in the SDK at all — the only per-turn stop is
`agent_max_turns`. §8.2.

### 3.5 What N agents can do under this design that one agent cannot

The value roast (§11.2) said, correctly, that every answer above is a
restriction. Restrictions are the substrate; they are not the point. This
section tests four capability candidates against the primitives and keeps
only the ones that survive. The rule for keeping one: it must be something
*one good agent in one context structurally cannot do*, not something it
merely does worse.

**1. Width over context — real, and already present in raw form.** A root
that fans out N briefs holds N *reports* in its context instead of N
*transcripts*. A single agent that reads 200 five-thousand-token support
threads does not have a context; it has a chunking problem, and every
chunk's tool output accumulates in the one context that has to make the
final call. `run_sub_session` already gives a single agent clean child
contexts today. What is new is not the width; it is that width is
**grantable**: today the only thing bounding a fan-out's spend is the
per-user in-flight cap, so the cap cannot be raised without unbounding
spend. Under a tree ledger, spend is bounded per tree regardless of the cap,
and the cap becomes what it should be — a load-shedding knob for the executor.
*Verdict: keep, stated precisely — the design does not create width, it
decouples width from spend so width can be offered.*

**2. Quarantine — real, and the one place monotone rules buy something.** A
child whose envelope carries `tools = {read_workspace_file}` and nothing
effectful can be pointed at untrusted input, and an injection in that input
has **no action surface**: there is no tool it can steer. The only thing it
can corrupt is the child's report — a typed, bounded, upward-flowing
artefact the parent reads as data. Compare one agent reading the page
itself: the injection steers that agent's *next tool call, mid-turn, before
any human sees anything*. Quarantine turns mid-turn injection into a
visible artefact. Two honest conditions: (a) the parent is **not** clean
afterwards — the report is attacker-influenceable content and must carry
the child's taint bit upward (taint flows down by inheritance and up by
report; what the quarantine confines is the *action surface*, not the
taint); (b) this only exists if the tools ceiling actually narrows, which
the roast showed is vacuous when the root has `permissions=None` — hence
§3.6's descent-denied default. **(c) — the sharp one, from T16:** "no action
surface" is a property of the *explicit read-only preset*
(`tools = {read_workspace_file}`, edge type #2), **not** of a default child.
A default delegate keeps `run_agent`, and `run_block`/`run_agent` reach ~78
outward blocks plus arbitrary HTTP via `SendWebRequestBlock`, none carrying
`is_sensitive_action`, in the one tier the auto-mode gate declines to judge
(T16, findings ledger). So the quarantine guarantee holds **only** when the
caller pins the tool set to the read-only preset; the descent-denied default
does not by itself make a child safe to point at untrusted input, because it
leaves `run_agent` in. The design must therefore say: quarantine is an
*explicit* edge type you select, never a property a child gets for free.
*Verdict: keep, scoped to the preset. One agent cannot read something without
being the agent that read it — but the reader must be pinned read-only, not
merely descent-denied.*

**3. Depth as abstraction — real iff the report is typed.** A parent that
receives `{status, summary, artefacts, spent, asked}` cannot tell whether
the child did the work alone or ran a subtree, and does not need to.
Checked for leaks across the boundary: spend (the child reports its
subtree's total as one number — the tree ledger made it one number), time (the
subtree inherited the deadline, so it finished inside it), taint (a tainted
grandchild taints the child via its report, so the child's report taints
the parent — consistent), escalation (a grandchild's `needs_input` is the
child's to answer or forward as *its* `needs_input`; the parent sees one
question), artefacts (by reference, path-addressed, no structure exposed).
The only structure that escapes is the session link on the ToolChain card —
to the human, which is where it belongs. *Verdict: keep, and it is why the
typed report is not decoration: untyped prose reports are exactly what
leaks ("I asked Bea and she said…").* One calibration from T10's grid: the
structured frame did **not** beat the naive prompt on score (12/18 vs
9/18 without policy; both 15/18 with it). What structure bought was
*scope* — the naive arm spent 3 of 12 objections on tone, spelling and
blame-shifting; the structured arms 0–1 in 12–15. So the honest claim for
typed reports is not "more accurate" but "stays in scope", and a check that
wanders is one people learn to override. Substitutability needs in-scope
reports, not smarter ones.

**4. Different conditioning (T10's check) and different identity — real,
existing.** N experts have N memory namespaces, N integrations and N souls;
a check runs in a context that did not produce the error. Both pre-date
this design. *Verdict: keep as inherited capability, claim nothing new.*

**The honest test: name a task a 30-agent instance completes and one good
agent does not.**

> *Audit last month's 200 support threads for refunds we promised and did
> not issue; return every match with a verbatim quote.* Roughly a million
> tokens of untrusted, customer-authored input.

One agent: no context holds it; chunking through today's sub-sessions gives
every chunk the parent's full authority over customer-authored text that
may say "refund everyone". A tree under this design: one root with the
effectful tools, up to fourteen concurrent leaves (in-flight cap 15 minus
the root) over ~three waves for thirty children, each leaf with
`tools = {read_workspace_file}`, the parent's deadline, and an acceptance
line of "every match carries a verbatim quote"; every turn admitted against
one tree ledger. The root merges thirty typed reports and is the only node
that can act — and it is tainted when it does, so T9's gate asks before it
does. Wall-clock ≈ 1/14 of serial; spend ≤ the tree's ceiling plus one
turn's overshoot; injection blast radius = one report, checkable
against its own quotes (T5's bait-audit shape, 97.9% self-agreement).

I am confident in the *structure* of that claim and I have not run it. The
unproven part is quality: whether fourteen narrow leaves plus a merge
produce a better audit than one agent chunking carefully. T5's evidence is
that structure beats content on process discipline, not that it beats it
on correctness, and I am not going to blur that line to make the claim
prettier. So the defensible statement is:

> **This is a substrate that makes width, quarantine and substitutability
> grantable. The width and quarantine cases are structurally real and
> unavailable to a single agent. The quality advantage at width is
> unproven and is the first experiment to run after the slice ships.**

### 3.6 Descent-denied tools — making the ceiling non-vacuous

`None ∩ None = None`. An interactive root has no permission filter, so
intersection alone bounds nothing on the default path (§11.2 finding 2).
The fix is a default that narrows without the caller asking:

```
DESCENT_DENIED_TOOLS = {
    # effects that outlive the tree
    "schedule_followup", "setup_agent_webhook_trigger", "update_preset",
    "store_skill",
    # effects that leave the platform or bind credentials
    "post_to_chat_platform", "connect_integration", "run_mcp_tool",
    # irreversible
    "delete_folder", "delete_preset", "delete_schedule", "delete_skill",
    "delete_workspace_file",
    # staffing (already denied to expert sessions; denied to every child)
    "hire_expert", "raise_expert", "update_expert", "confirm_expert_change",
}
child.tools = (parent.tools or ALL) ∩ (request.tools or (ALL − DESCENT_DENIED))
```

A parent *may* grant a denied tool explicitly — a root that wants a child to
post to Slack says so in the request (`grant_tools` on all three spawn
tools) — and the grant is still intersected with what the parent holds, so
it never widens. What changes is the default: **a child, unless told
otherwise, cannot create persistent, outward or irreversible effects.**

Memory writes are governed by a second, narrower rule keyed on the
*namespace*, not on descent: a child that shares its spawner's namespace
(an isolate — same `expert_id`, or both AutoPilot) loses `memory_store` /
`add_understanding`, because what it stores the spawner reads back next
turn as its own memory (§11.1 finding 1). A delegated expert writes to its
own namespace and keeps them — storing what it learned is its normal
behaviour, and T9's taint rule governs the tainted case. The first version
of this section put memory writes in the descent-denied set outright; the
compatibility review (§12) showed that would silently stop every delegated
expert from learning, which the laundering argument never required.

---

## 4. The invariants, and where each is enforced

| # | invariant (from the brief) | holds by | enforcement seam |
|---|---|---|---|
| 1 | Authority never widens on descent; cannot be obtained from a third party | `tools = (spawner or ALL) ∩ (request or ALL − DESCENT_DENIED)`; the envelope is derived from the **running turn's** envelope in the executor's contextvar, never from a row or from the model; there is no tool that grants permissions; a resumed child runs under the caller's *current* envelope; the queued-into-in-flight path is refused for spawns; `expert_admin` is denied to every expert session (exists) | `derive_child_envelope` at `schedule_turn`; **`BaseTool.execute` refuses a tool outside the turn's set on both engines** (hiding in the schema is not enforcement — §11.1 finding 5); `execute_tool` re-checks groups (exists) |
| 2 | Budget is bounded per tree: a turn is admitted iff the tree's metered spend is under its ceiling; Σ tree ≤ ceiling + one turn's overshoot | one Redis key per tree, Lua `admit` at turn start, `charge` on turn end from the existing cost-recording path; fail closed for depth > 0 when Redis is unavailable | `schedule_turn`; `token_tracking.py` |
| 3 | Taint propagates downward and never launders; and upward by report | `tainted = spawner.tainted or born`; a new session id no longer means a clean bit because the bit is on the turn, not the session; a report carries its author's bit | `derive_child_envelope`; report model; T9's `is_tainted()` reads `envelope.tainted` as a source |
| 4 | Locality | every bound is a field on the turn or one key per tree; no fan-in, no traversal, no global set; leaves get no roster; spawning nodes get a pod-scoped one | by construction; `_team_context` |

**Restated invariant 2, on the roast's argument (§11.1 finding 17):** the
first draft claimed strict sub-allocation — Σ subtree ≤ root by
conservation. The executor cannot enforce that: the SDK's `max_budget_usd`
is a per-query stop, floored upward to $0.50, re-issued per stream attempt,
and absent on the Codex transport. A conservation law the platform cannot
enforce is worse than an honest counter, because it invites the UI to
display the number as a bound. The accounting form — one metered counter,
checked at the one seam every turn passes through — is what actually holds.
Argued explicitly since the brief called it a hard invariant: the *intent*
(spend at scale is bounded per tree by its root's allocation, with no global
accounting) is fully preserved; only the *mechanism* changed from
reservation to metering.

**A note on invariant 3 as it stands on `dev`:** nothing on `dev` *sets*
taint, because T9 is a branch. This plan builds the carrier and the
propagation and one birth source (chat-platform sessions, `source_platform`
set — T9 §5.2), so that when T9 lands the laundering path is already closed
and its gate has one more boolean to read. The claim I can make on `dev`
alone is "taint, once set, cannot be laundered through a spawn"; the claim
"taint is set when it should be" is T9's.

**The two channels the envelope does not close, named (§11.1 findings 1, 2):**
memory is shared *along identity* (an isolate shares its parent's
namespace; every AutoPilot node shares `user_<id>`), and the workspace is
user-wide. Neither is a tree property and neither can be made one without
rebuilding the storage layer. The design closes the **write** side for
children instead: memory/skill writes are descent-denied by default, and
workspace writes at depth > 0 are confined to the writer's own session
folder. A child can still *read* what its identity or its user can read —
that is data by reference, and it is what artefacts are.

---

## 5. Where T10 and T9 slot in

**T10's `consult_teammate`** is edge type #1 (§2.6): an envelope with no
spawning, `tools = ∅`, one bounded completion charged to the tree, no
session, and a brief/report that the general schema expresses exactly once
`content` and `findings` were added. The pod reviewer becomes "the expert
the pod's members should consult before committing", exactly as T10 had it.

**T9's gate** reads `envelope.tainted` as a taint source alongside its
transcript scan, and — this is the part that matters at scale — its
"delegation is ALWAYS_ASK" rule can be relaxed to "delegation is ALWAYS_ASK
*when the child would be granted anything the gate would ask about*". A leaf
child pinned to the read-only quarantine preset (edge type #2), with a
2-minute deadline and a tree already under a ceiling, does not need a human
to approve it; the envelope is the approval. **But note the T16 limit above:
a merely descent-denied child still holds `run_agent`, which reaches
outward blocks and arbitrary HTTP the gate does not judge — so the "envelope
is the approval" shortcut applies to the read-only preset, not to a default
child.** That is the ergonomic win the two designs get by composing, scoped
honestly: T9's gate stops asking about the spawns the envelope has actually
made safe, and keeps asking about the rest.

Neither integration is built here. Both are one-boolean / one-field seams.

---

## 6. Failure modes

| failure | behaviour |
|---|---|
| Child crashes mid-turn | Existing: stream error → `failed`. Its recorded cost (if any) is charged to the tree by the existing usage path; nothing was reserved, so nothing leaks. |
| Parent fans out past the per-user in-flight cap (15) | Existing: `rejected_concurrent_turn_cap` — the spawn fails before any side effect; the tree's `nodes` counter is decremented (admit → schedule → on failure, release). Children are *rejected*, never queued behind a parent that is waiting on them. |
| Target session already has a turn in flight (resume of a prior delegation while the user is typing in it) | **Refused** (`target_busy`). The first draft's mechanism would have appended the child's prompt into the running turn under that turn's permissions (§11.1 finding 3). |
| Two parallel spawns race | Both hit the tree ledger's Lua `admit`; Redis serialises them; the second either fits or is refused. No row is touched. |
| A node tries to resume, poll or cancel a session it did not create | Refused: the creator capability (`delegated_by_session_id`) does not match. Removes self-resume and sibling-resume, which were the roast's self- and mutual-deadlock (finding 11). |
| Deadline passes while the child is running | **New** watcher (not built in the slice): a per-tree expiry on the ledger key plus a scan that fires `enqueue_cancel_task`; child finalises `failed`; report `out_of_time`. Today the only callers of cancel are the stop button and `get_sub_session_result(cancel=true)`. |
| Child asks a question | `needs_input` up the tree; the root turns it into `ask_question`. Known caveat inherited from `model.py::clear_pending_question`: a machine-injected user-role turn clears a pending question the human never answered. |
| Tree exhausts its ceiling mid-tree | The next `admit` in that tree refuses; the refusing spawn returns `out_of_budget` to its caller; the root's own turn continues (its admission already happened) and reports honestly. |
| Legacy session, no provenance | It can only be a root (roots have no provenance). A legacy *child* row without `delegated_by_session_id` (a pre-change `run_sub_session` sub) cannot be resumed under the new creator rule — a conscious break: resuming it would be resuming a sub whose creator is unknown. |
| Redis unavailable | Spawns (depth > 0) fail closed; root turns behave as the per-user rate limit already does. The first draft's "nothing here lives in Redis" was false: session reads, in-flight detection, the pending buffer and the spend counters all do. |
| Expert weekly budget breached by one tree's graph run | That expert's schedules pause for **every** tree — a deliberately non-local, per-identity guardrail the owner configured. Unchanged; see §11.1 finding 13. |

---

## 7. What I am NOT building

- No orchestrator, planner, scheduler, router, or manager expert.
- No message bus, task board, work-item model, DAG, or queue beyond the one
  that exists.
- No lateral edges: no sibling, peer, or broadcast communication.
- No shared memory, shared session, or team-wide context. Memory scoping is
  untouched.
- No voting, debate, consensus, or multi-round negotiation.
- No budget *reservation* or lease return — the ledger meters; nothing is
  reserved, so nothing has to flow back.
- No per-file workspace capabilities — the storage layer is user-scoped;
  children get write confinement instead.
- No memory isolation along the tree — memory is identity-bound and stays
  so; children lose the write side by default.
- No TEAM/ORG-visible experts — tenancy work, orthogonal.
- No T9 gate and no T10 consult — both slot in; neither is duplicated.
- No new expert kind, no new UI surface beyond an envelope line on the
  existing delegate ToolChain card.
- No cross-user spawning of any kind.

---

## 8. What cannot be made to work as specified, honestly

1. **"3000 agents" as 3000 concurrently running nodes under one owner** does
   not work and should not: the per-user caps exist to protect the executor
   and the user's wallet. The design is scale-invariant in *rows* and in
   *trees per user*, not in concurrent sessions per user.
2. **Budget is metered, not reserved, and the overshoot is coarse.** A turn
   admitted under the ceiling may spend up to the SDK per-query cap
   (floored at $0.50, up to three stream attempts) before the charge lands
   — the same order as the default $1.00 daily cap. On the Codex transport
   there is no USD cap at all and the only per-turn stop is
   `agent_max_turns`. Graph runs a child starts with `run_agent` are
   charged in credits to the user's wallet and the expert's weekly budget
   — **not** to the tree ledger — in v1. Naming it rather than hiding it:
   the ledger bounds LLM spend across the tree; the wallet and weekly
   budget bound graph spend; the two are not yet one number.
5. **Memory and workspace are not tree-scoped and this design does not
   make them so.** Children lose the write side by default; a child that is
   explicitly granted `memory_store` by a parent that holds it can still
   write into a namespace its parent reads next turn. The grant is the
   parent's decision and it is visible in the envelope; it is not a
   laundering path the platform opened by itself.
6. **A descent-denied child is not an outward-safe child.** The default
   narrowing removes the named outward tools, but it leaves `run_agent`,
   and `run_block`/`run_agent` reach ~78 outward blocks plus arbitrary HTTP
   via `SendWebRequestBlock`, none flagged `is_sensitive_action` (T16). So
   the only child that is genuinely safe to point at untrusted input is one
   pinned to an explicit read-only tool set (edge type #2), never one that
   merely inherited the descent-denied default. The design states this as a
   rule (§3.5) rather than letting a reader assume the default is a
   sandbox.
3. **A brief's `authority` line is an assertion by the parent, not a proof.**
   A tainted parent can write a false authority line. The taint bit is what
   tells the child (and T9's gate) not to trust it — which is why taint and
   authority travel in the same envelope and why nothing here lets a child
   act on authority alone when tainted.
4. **Process, not correctness.** Acceptance lines and reports are auditable
   for discipline (T5), not for truth. A parent cannot tell from a report
   that the child was *right*. Nothing in this design claims otherwise, and
   anything that did would be the rubric that T5 round 1 buried.

---

## 9. The build slice

Revised after both roasts. The value roast said the first slice was the
easiest one, not the one that demonstrates the claim; the architecture
roast said the mechanism it would have built cannot bind. The core claim is
now: **every turn in a tree passes one chokepoint that derives its envelope
from the spawning turn and admits it against the tree's ledger, and the
tools in that envelope are refused — not hidden — on both engines.** The
slice that demonstrates it:

1. `copilot/tree.py` — `TurnEnvelope`, `SpawnRequest`, `DESCENT_DENIED_TOOLS`,
   `derive_child_envelope` (pure), `root_envelope`, `TreeRefusal`, and
   `TreeLedger` (`open`, `admit`, `release`, `charge` — one Redis key per
   tree, Lua-atomic, fail-closed for depth > 0). Tested in `tree_test.py`
   with a fake Redis: monotonicity of every field, refusal on every
   overflow, admit/charge accounting under concurrent admits.
2. `CoPilotExecutionEntry.envelope`; the executor's `set_execution_context`
   carries it; `get_current_envelope()` next to `get_current_permissions()`.
3. `schedule_turn` derives the child envelope from the caller's current one
   (or opens a root), admits against the ledger, releases on dispatch
   failure, and puts the envelope on the entry.
4. `BaseTool.execute` refuses a tool outside the turn's envelope. Both
   engines, all registry tools, one place.
5. `run_copilot_turn_via_queue(allow_queue=False)` for every spawn and
   resume: a busy target is `target_busy`, never a buffer append.
6. `run_sub_session` writes `delegated_by_session_id`; its resume and
   `get_sub_session_result`'s same-scope arm require the creator
   capability.
7. `token_tracking` charges the tree.
8. `_team_context` is skipped for leaf turns.

A demo script (`scripts/`, not pytest) runs a fake fan-out of 30 children
under one root against the fake Redis: asserts every child's tools ⊆ root −
descent-denied, depth never exceeds 3 through an isolate → delegate →
isolate chain, the 9th node is refused at `max_nodes = 8`, and a tree whose
metered spend crosses its ceiling admits no further turn. No LLM is called;
the claim is structural and the demonstration is of the structure.

**What was built (commit `5422a83e99`).** Items 1–7 above, with these
specifics worth knowing before reading the diff:

- `copilot/tree.py` (envelope, derivation, ledger, admit/release/charge)
  and `tree_test.py` (22 tests: monotonicity of every field, refusal on
  every overflow, concurrent admits under a fake Redis, fail-closed for
  children / fail-open for roots when Redis is down). Run without pytest
  via a plain runner per the brief's rule; 22/22 pass. The demo prints:

  ```
  fan-out: 7 admitted, 23 refused          (max_nodes 8, root counted)
  every leaf: tools=['read_workspace_file'], cannot spawn, cannot act outward
  hop 4 (delegate) refused at depth 3
  a child asking for a tool its spawner lacks does not get it
  after 5 charged turns the tree refused    (ceiling $0.50, $0.12/turn)
  ```

- The ledger admits with increment-then-check-then-rollback on the node
  counter — the same non-locked pattern `acquire_turn_slot` uses — rather
  than Lua; the over-admit-by-one under a race is the bound's stated slack.
- Fan-out is counted **per tree** (`tree_max_nodes`, default 8), not per
  node per turn, because only the former is representable atomically.
- All three spawn kinds may spawn onward within `MAX_DEPTH = 3`. The first
  cut made isolates leaves; the compatibility review (§12) showed that
  would refuse a sub-AutoPilot that needs a teammate, which is a working
  flow today. Isolates now count toward depth (they did not) and lose
  memory writes (they share the namespace); leaf-ness is the quarantine
  preset (`tools=[…]`), not the default.
- `grant_tools` on all three spawn tools is how a spawner hands down a
  descent-denied tool it holds — the ToolChain card shows the grant at the
  one place a human can see it.
- `run_sub_session` now writes `delegated_by_session_id`, which also means
  an isolate's `pending_question` no longer surfaces on Home (the Home
  predicate excludes delegated sub-threads) — consistent with the tree
  rule that a child escalates to its parent, and noted here because it is
  visible.
- `chain_refusal`'s provenance walk is left in place (loop check plus
  belt-and-braces on depth); the envelope is the real bound.
- Existing `sub_session_test.py` fixtures were updated for the creator
  rule and two tests added (resume requires the creating session; a fresh
  sub records its creator). Other suites were not run (brief: no pytest);
  `dispatch_turn` touches Redis only for spawned turns (depth > 0), so
  every existing root path — HTTP chat, scheduler, `AutoPilotBlock`, and
  their tests — is byte-for-byte unaffected by the ledger.

Not in the slice, each a named follow-up: deadline watcher; brief slots in
the tool schemas (`content`, `authority`, `acceptance`) and typed report
parsing (`findings`); workspace write confinement; `block_filters` (tools
only in v1); roster withheld from leaves and pod-scoped for spawning nodes;
T9/T10 integrations.

---

## 10. Sensitive findings — where they are described

Two of §1.3's grounding facts are logged in `~/code/agpt/.claude/log/findings.jsonl`.
§12.1 traces both and downgrades them the way T9's were: Fact A is a missing
control T9's gate needs, not an exploitable widening; Fact B is a churn
guardrail gap capped by the rate limits, not a new bad outcome. They are
described in this file only as facts and design consequences — §1.3 (two
bullets), §3.2 (the `tools` and `depth` rows), §4 (invariants 1 and 3), §12.1.
No worked sequence appears anywhere in this repository.

## 11. The roasts

Two sub-agents, in parallel, different jobs. Both were given the first
committed draft of this file (`8bcec639e4`) and the code and told to break
it, not improve it. Their findings are summarised below with what changed
and what I rejected. Where a finding described a concrete misuse path in
shipped code, the path is not reproduced here (§10).

### 11.1 Architecture roast — 17 findings, 4 CRITICAL

**Verdict as delivered:** "The design does not survive as written. Its
central claim — the envelope is the child's *sole* connection to its parent
— is falsified three times over by channels that already exist and that the
plan never inventories." The single recommended change: model the bound as
a per-tree record on the *turn*, enforced at `schedule_turn`, failing closed
on the queued path.

**Accepted and rebuilt (the mechanism):**

- *F1 CRITICAL — memory is a shared writable channel.* Isolates share
  their parent's namespace; every AutoPilot node shares `user_<id>`. A
  child's `memory_store` is read by the root next turn. → Memory is named
  as identity-bound and **not** tree-isolated (§4); memory/skill writes are
  descent-denied by default (§3.6).
- *F2 CRITICAL — the workspace is user-flat with cross-session write.*
  "READ capability on exactly these" was unimplementable. → Artefacts are
  references, not capabilities; children write only inside their own
  session folder (§2.4).
- *F3 CRITICAL — the queued-into-in-flight-turn path drops permissions and
  runs a child's prompt inside another session's turn.* → Spawns and
  resumes never queue; a busy target is refused (§2.3).
- *F4 CRITICAL — `run_sub_session` resume is an unguarded lateral edge, and
  "origin = automation always" would have widened it.* → `run_sub_session`
  writes provenance; resume/poll/cancel require the creator capability;
  the origin change is dropped — taint is its own field (§2.3c).
- *F5 HIGH — the baseline engine only hides tools; `execute_tool` never
  checks permissions.* → Enforcement moves into `BaseTool.execute`, one
  place for both engines (§4).
- *F6 HIGH — no `blocks` in the envelope; `_parent` is dropped on the
  queue.* → Serialisable `block_filters` list (§2.3a); tools-only in the
  slice, named.
- *F7, F8, F9 HIGH — `max_budget_usd` is per-query, floors up to $0.50,
  re-issues per attempt, is absent on Codex, and the $1.00 default daily
  cap makes a $2 lease tree arithmetically impossible.* → The lease is gone.
  One metered counter per tree, checked at turn start (§2.3b); the SDK cap
  becomes a per-turn stop, not the invariant.
- *F10 HIGH — a CAS on a cached, immutable, in-memory session row is not a
  CAS; per-turn fan-out is not representable there.* → State split by
  lifetime: envelope on the turn, ledger on the tree in Redis, provenance on
  the session (§2.3).
- *F11 HIGH — self- and mutual deadlock via the ride-the-in-flight path.* →
  Closed by F3 + F4.
- *F12 MEDIUM-HIGH — spawn is O(N): every new session injects the full
  roster with workflows joined.* → Leaves get no roster; not deferrable
  (§3.4).
- *F14 — wrong "exists" claims:* spawn paths use the in-flight cap (15) not
  the running cap (5); there is no deadline watcher, only a cancel function;
  `list_sub_workspace_files` enumerates, it does not scope. → All corrected
  in place.
- *F15 MEDIUM — handoff + origin=automation stamps the human's thread as
  machine-driven, and a "converted root lease" either starves or mints.* →
  Origin change dropped; the handoff turn is the parent tree's last node and
  later human turns are new roots (§2.1, §3.3).
- *F16 — missed failure modes:* resume of a prior delegation bypassed the
  envelope (→ the envelope is per turn, so a resumed child runs under the
  caller's current one); legacy fail-open depth (→ roots have no
  provenance; legacy children without it cannot be resumed); injected turns
  clear `pending_question` (→ named in §6); "nothing lives in Redis" was
  false (→ §6 corrected, fail-closed rule stated).
- *F17 — invariant 2 is wrong as stated.* → Accepted and restated as an
  accounting invariant (§4), with the argument for why the brief's intent
  survives the restatement.

**Rejected, with reasons:**

- *F13 MEDIUM — the expert weekly budget is a cross-tree kill switch,
  "contradicting §3.2's own rationale".* Not contradictory: the weekly
  budget is a per-identity guardrail the expert's owner configured, and a
  breach pausing that identity everywhere is what "this expert may not
  spend more than X/week" means. It is non-local across *trees* but local
  to the *tenant*, which is the locality that matters at 3000. The finding
  is right that my §3.2 sentence about "draining a frugal expert's week"
  described the current behaviour rather than a hypothetical; the sentence
  is reworded, the mechanism stays.

### 11.2 Value roast — 12 findings, 2 KILLS-IT

**Verdict as delivered:** "Not yet more valuable than one good agent, and
the document does not claim it is … a well-argued monotone capability
envelope … every answer here is a narrowing." The one addition that would
change the answer: a concurrent fan-out primitive with a demonstration that
total spend ≤ root *and* a wall-clock win. The one deletion: `authority` as
a required field.

**Accepted:**

- *F1 KILLS-IT — all bounds, no new capability; T10's check apparently
  deleted.* Half accepted. The consult edge was always kept (§2.2); "not
  building" meant not re-implementing. But the substantive point stood and
  §3.5 now exists because of it: four capability candidates tested against
  the primitives, two kept as structurally new (width made grantable;
  quarantine), one kept conditional on typed reports (substitutability),
  a named task, and the honest statement that quality-at-width is unproven.
- *F2 KILLS-IT — the tools ceiling is vacuous on the default path
  (`None ∩ None`).* Fully accepted; the most important single finding of
  either roast for the security story. → `DESCENT_DENIED_TOOLS` (§3.6):
  children lose persistent, outward and irreversible effects by default; a
  parent may grant only what it holds.
- *F3 MAJOR — the typed brief is `system_context` with subheadings; the
  real mechanism (artefacts by reference) is not built.* Accepted that the
  slots are prompt structure, not enforcement, and the plan now says so.
  The artefact half met the architecture roast's F2 coming the other way:
  the storage layer cannot back a per-file capability, so artefacts are
  references and the write side is confined.
- *F5 MAJOR — "no monotone answer" for lateral edges is false.* Accepted;
  §3.3 now gives the real reasons (cycles → deadlock; doubled surface).
- *F6 MAJOR — lease floor starves the parent; argues lease return into v1.*
  Dissolved rather than accepted: with metering there is no reservation, so
  a parent is never starved by children's *allocations*, only by their
  *spend*. The finding was correct about the lease design; the lease design
  is gone.
- *F7 MAJOR — envelope immutability contradicts CAS; in-memory fan-out
  count is not a bound.* Accepted (same as architecture F10).
- *F8 MAJOR — `authority` as a required slot is a schema tax.* Accepted;
  optional. T10's later measurement (6/18 on an identical prompt) is why it
  stays a slot at all.
- *F9 MAJOR — §9 was the easiest slice, not the demonstrating one.*
  Accepted; §9 rewritten around the chokepoint and a 30-node fan-out demo.
- *F10 MAJOR — the plan never says when a spawn beats one context or prices
  one.* Accepted in substance via §3.5; the price is stated there (a full
  system prompt, first-message context, a blocked parent holding a slot up
  to 300 s) and the honest sentence — outside width, quarantine, different
  identity and different conditioning, delegation loses — is now in the
  file.
- *F11 MINOR — cap numbers wrong.* Corrected.

**Rejected, with reasons:**

- *F4 MAJOR — "the useful middle is answered 'the caller decides', which
  is what exists."* Partly. The shape (per edge, typed, one-directional) is
  the answer to "one setting or per relationship", and it is a real
  answer. What I concede is the substance: there is no *mid-turn pull* — a
  child that discovers it needs more must end its turn with `needs_input`.
  That is not a dodge but a structural fact: the parent is blocked inside a
  tool call while the child runs and cannot service a question without a
  turn boundary, and a lateral pull would be the cycle §3.3 rejects. The
  default guidance is now explicit: the brief carries what the child needs
  to finish without asking; `needs_input` is the retry path.
- *The proposed "concurrent fan-out primitive".* Not added as a new tool,
  and this is a deliberate rejection: parallel fan-out already exists —
  `run_sub_session(wait_for_result=0)` returns immediately and the CLI
  dispatches registry tools concurrently (`readOnlyHint=True` on all of
  them), so a parent already can launch N children and join them with N
  polls. What was missing was not the primitive but the *bound* that makes
  it grantable at width, and that is what the ledger is. Building a second
  fan-out tool on top of a working one is the orchestration-framework trap
  T10 correctly refused.

---

## 12. Reachability and compatibility — checked before pushing

### 12.1 Are the two grounding facts exploitable on `dev` today?

The orchestrator asked the question T9's retraction taught: a finding is a
hypothesis until someone traces reachability. Traced against `dev`:

**Fact A — delegation passes permissions equal to the parent's, with no
narrowing path.** For an untrusted input to *gain* something through a
delegation, the child would have to hold a capability the parent lacked.
It does not: `permissions` is copied verbatim, tool *groups* are derived
from `expert_id` and the child's group set is never a superset of a
capability the parent held in substance (a plain session's `update_expert`
covers an expert child's `update_expert_soul`; `handoff_to_expert` is a
variant of `delegate_to_expert`), and the staffing guard is *stricter* for
children (automation origin). On `dev` an interactive parent is already
ungated on every outward tool (`post_to_chat_platform`, `schedule_followup`,
`bash_exec` … — `requires_auth` is the only check), so there is no gate for
a fresh session to launder past; that gate is T9's, on a branch. What a
delegation adds is the *target identity's* memory namespace and voice —
the same user's data, a confidentiality boundary between that user's own
experts, not a privilege boundary. The `_parent` PrivateAttr loss on the
queue is moot in practice: `AutoPilotBlock`'s inherited-permission
contextvar only survives in-process, and nested graph runs execute in
separate node executions. **Verdict: not exploitable. A missing control
that T9's gate needs to exist, and a design gap — not a vulnerability.**

**Fact B — `run_sub_session` does not count toward the depth bound.** The
bad outcome the bound exists for is a chain sustaining itself on credits.
Resetting depth buys a longer cross-expert chain, but a same-identity
isolate can already recurse *without any bound* today, so an agent that
wanted to spend did not need the reset — and both are stopped by the
per-user in-flight cap (15) and daily/weekly cost caps, which apply
regardless of depth. The loop check (`seen` set) still holds for
cross-expert hops. **Verdict: no new bad outcome reachable. A churn
guardrail gap — not a vulnerability.**

Both findings therefore downgrade the same way T9's did, and the push is
clear on that basis. Their ledger entries should be annotated accordingly.

### 12.2 What this slice changes for flows that work today

Stated first, before any capability claim, because each one is a decision
rather than a bug:

The default is split into two tiers so the break surface is as small as the
security goal allows:

| shipped flow | before | after | judgement |
|---|---|---|---|
| A delegated / handed-off expert **posts to a chat platform, schedules a follow-up, calls an MCP tool, stores a skill, or stores memory** | inherited from the parent | **still works** — a delegate runs under its own identity, namespace and budget, so these are its job and stay in the default | No change. The scheduled-standup post (the `chat_platform` headline use case) and MCP-integration delegations are unaffected. |
| A delegated / handed-off expert **connects an integration, sets up a webhook trigger, edits a preset, or deletes a folder/preset/schedule/skill/file** | inherited | **refused unless the spawner passes `grant_tools`**; the refusal names the tool and tells the child to report the need | Intended. These bind the account or are irreversible and are not a normal delegate job. Rare, visible, recoverable — never silent data loss. |
| A `run_sub_session` **isolate** posts outward, schedules, calls MCP, stores a skill, or **writes memory** | inherited | **refused unless granted** | Intended, and the change most likely to be noticed. An isolate shares the parent's identity and memory namespace, so a write is the parent's write (finding 1) and a post is under the parent's name. This is the shared-namespace leak the design exists to close. |
| A chain that used `run_sub_session` isolates to go **past three hops** | unbounded via isolates | **stops at depth 3** (isolates now count) | Intended. Isolates keep `may_spawn=True`, so a single isolate under a shallow chain is unaffected; only chains implicitly deeper than three via isolates are cut. The bound was always meant to be total. |

The escape hatch for every refusal is one field — `grant_tools` on
`delegate_to_expert` / `handoff_to_expert` / `run_sub_session`, intersected
with what the caller holds so it can never widen. Reinier decides whether the
isolate memory-write default is too aggressive for `dev`; flipping it is
removing two entries from `ISOLATE_DENIED_TOOLS`.

