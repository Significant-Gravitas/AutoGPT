# Agent collaboration architecture (TODO T15)

**Status:** plan → two parallel roasts (§11) → smallest working slice (§9).
Branch `pwuts/agent-collab-architecture`, worktree `AutoGPT4`.

---

## 0. The answer, up front

T10 said *a team is a check*. That is true and it is the wrong unit. The unit
that scales is not the team and not the check; it is **the edge** — the thing
that is created when one agent causes another to exist or to act. Every hard
question in the brief (coupling, inheritance, position, scale) is a question
about what an edge carries, and every invariant is a statement that something
carried across an edge is monotone.

So the design is one primitive and four uses of it:

> **Every spawn creates an Envelope.** The envelope is computed *only* from
> the parent's envelope and the spawn request, by operations that can only
> narrow (intersection, minimum, subtraction, logical-or of a taint bit). It
> is the child's sole connection to its parent. Nothing else crosses.

The four questions, in one line each:

| question | answer |
|---|---|
| **Context coupling** | Per edge, not global — but the *shape* is fixed by the platform and the *content* is chosen by the caller. What crosses is a **brief** (task, artefacts by reference, claimed authority, acceptance) and a **report** back. Transcript, memory and tool results never cross. |
| **Inheritance** | Split into three classes: **ceilings** (permissions, budget, deadline, depth, fan-out, dry-run, origin, taint, tenancy, billing route) inherit monotonically and can only narrow; **identity-bound** state (soul, memory namespace, voice, workflows, the expert's own weekly budget) is fixed by *which* expert the child is, never by the parent; **free within ceiling** (model, effort, wait time) is the caller's choice and is bounded by budget, not by tier. §3.2 is the table. |
| **Position** | A child has exactly two edges: in from its parent (brief) and out to its parent (report + progress). Nobody else can address it; it can address nothing but its parent and its own children. Siblings share data only through artefacts the parent hands both. `handoff` re-roots a node so the human becomes its parent. |
| **Scale** | The graph is a forest of trees rooted in *turns*. Every bound is per-node or per-edge and is checked at spawn from O(depth) rows. Budget is a **lease tree** (conservation, not accounting). The one non-local thing today — the roster injected into every prompt — becomes scoped addressing plus a directory lookup. No agent ever sees the whole graph. |

And the thing I am not going to pretend: at N = 3000 the *concurrency* limits
are per user (running 5 / in-flight 15) and they are the actual bound on how
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
with a fresh lease. This is what makes the whole thing scale-invariant: no
tree outlives the turn that rooted it, and no bound has to be reconciled
across turns.

### 2.2 Edge kinds

Four, and only four. Three already exist; the fourth is T10's and slots in
as a degenerate case.

| kind | tool | identity of child | what returns | may spawn | who the child escalates to |
|---|---|---|---|---|---|
| **delegate** | `delegate_to_expert` | the target expert | report, pollable by the parent | yes, within envelope | its parent |
| **isolate** | `run_sub_session` | same as parent | report, pollable by the parent | yes, within envelope | its parent |
| **handoff** | `handoff_to_expert` | the target expert | nothing (`transferred`) | yes, within envelope | **the human** — the node is re-rooted |
| **consult** | T10 `consult_teammate` | the target expert's soul only | a typed verdict, inline | **no** — no tools, no session | n/a (stateless) |

`handoff` is the only edge that changes who a node reports to, and it does so
in exactly one direction: toward the human. There is no edge that makes a
node report to a sibling, a peer, or a third expert.

### 2.3 The Envelope — the invariant carrier

New typed fields on `ChatSessionMetadata` (no migration; defaults for legacy
rows), written once at session creation and immutable after:

```python
class Envelope(BaseModel):
    root_turn_id: str           # the tree this node belongs to
    depth: int                  # 0 for a root
    children_remaining: int     # fan-out budget for THIS turn
    budget_microdollars: int    # lease carved from the parent at spawn
    deadline_at: datetime       # min(parent.deadline_at, now + requested)
    tainted: bool               # parent.tainted OR born-tainted
    tools: frozenset[str] | None  # None = parent was unrestricted; else ⊆ parent
```

plus the fields that already exist and belong to the same object in spirit:
`origin`, `dry_run`, `llm_auth_provider`, `llm_credential_id`,
`delegated_by_session_id`, `handed_off_from_expert_id`, and the session's
`organization_id` / `team_id` / `user_id`.

**The derivation is one pure function** and it is the whole security argument:

```
derive_child_envelope(parent: Envelope, req: SpawnRequest) -> Envelope | Refusal
  depth              = parent.depth + 1            ; refuse if > MAX_DEPTH
  children_remaining = req.max_children            ; refuse if parent.children_remaining == 0
                                                   ; parent.children_remaining -= 1 (persisted CAS)
  budget             = min(req.budget, parent.remaining_budget)
                                                   ; refuse if < MIN_FUNDABLE (never floor up)
                                                   ; parent.remaining_budget -= budget (persisted CAS)
  deadline_at        = min(parent.deadline_at, now + req.max_seconds)
                                                   ; refuse if already past
  tainted            = parent.tainted or req.born_tainted
  tools              = parent.tools ∩ req.tools    (None ∩ X = X; X ∩ None = X)
  origin             = "automation"                ; ALWAYS, for every child
  dry_run            = parent.dry_run or req.dry_run
  billing route      = parent's, unchanged         ; refuse a request to change it
  tenancy, user_id   = parent's, unchanged
```

Every operation is monotone in the direction the invariants require. There
is no field a request can *raise*. A child cannot ask a third party either:
the only code path that creates a child session reads the *parent's*
persisted envelope from the DB, not anything the child or the model supplies.
That closes permission laundering by construction rather than by policy.

Two of the operations are **CAS writes to the parent's row** (fan-out and
budget). They are the only state mutated at spawn, they touch exactly one row
(the parent), and they are what make the bounds hold under parallel tool
dispatch — the SDK marks every MCP tool `readOnlyHint=True` so the CLI *will*
fire two `delegate_to_expert` calls concurrently.

### 2.4 The Brief — what a parent chooses to send

This is the context-coupling knob, and it is per edge. It is *not* free text
with a system-context prefix (what exists today); it has slots, and the slots
are the T10 lesson turned into a schema:

```
SpawnRequest:
  task: str                     # what to do, written for someone who cannot see this thread
  artefacts: list[WorkspaceRef] # files by id — the child gets a READ capability on exactly these
  authority: str                # what the caller asserts the child may assume/commit to  (T10)
  acceptance: str               # what "done" looks like, in a form the child can check itself against
  max_children: int = 0         # fan-out grant — default: the child is a leaf
  budget_microdollars: int      # lease request — clamped, never raised
  max_seconds: int              # deadline request — clamped, never raised
  tools: list[str] | None       # narrowing request — intersected, never widened
```

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
  artefacts: list[WorkspaceRef]   # what the child wrote, by id (exists: list_sub_workspace_files)
  spent_microdollars: int         # for lease return
  asked: str | None               # the question, when status is needs_input / needs_approval
```

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
| `user_id`, `organization_id`, `team_id` | ceiling | identical | ✔ | `create_chat_session` copies from parent; child cannot pass its own |
| billing route (`llm_auth_provider`, `llm_credential_id`) | ceiling | identical | ✔ | exists: resume checks refuse a mismatch; spawn copies |
| `dry_run` | ceiling | `parent or requested` | ✔ (true is sticky) | exists for delegate/isolate; make it the envelope rule |
| `origin` | ceiling | `"automation"` for every child | ✔ (never regains `interactive`) | **change**: `child_session_origin` currently copies the parent's |
| taint | ceiling | `parent or born_tainted` | ✔ (never clears) | new field; T9's gate reads it as a source |
| tools / blocks | ceiling | `parent ∩ requested` | ✔ | new: spawn intersects; executor applies `apply_tool_permissions` (exists) |
| budget (lease) | ceiling | `min(requested, parent.remaining)`; parent.remaining −= child | ✔ (sum of subtree ≤ root) | new: CAS on parent row; child's SDK `max_budget_usd = min(dynamic, lease)` |
| deadline | ceiling | `min(parent.deadline, now + requested)` | ✔ | new field; executor cancels at deadline via `enqueue_cancel_task` (exists) |
| depth | ceiling | `parent + 1 ≤ MAX_DEPTH` | ✔ | new field replaces the provenance walk; applies to isolate too |
| fan-out | ceiling | child gets `requested`; parent's count −1 | ✔ | new: CAS on parent row |
| escalation target | structural | parent; the human only at a root; handoff re-roots | ✔ | exists (Home predicate) + report status |
| identity (soul, name, voice) | identity-bound | the *target expert's*, from its row | caller cannot inject into the system prompt | exists: suffix built from the expert row; `safe_caller_name`; `fence_voice_preferences` |
| memory namespace | identity-bound | the target expert's | not the parent's, not the task's | exists: `derive_memory_group_id(session.expert_id)` |
| workflows, integrations | identity-bound | the target expert's | — | exists |
| expert weekly budget | identity-bound, orthogonal cap | the target expert's own counter | — | exists: `enforce_expert_run_budget` on graph runs |
| model, thinking effort | free within ceiling | caller's choice, routed by `resolve_model_route` | — | budget is the bound, not tier |
| wait time (`wait_for_result`) | free within ceiling | caller's choice ≤ `MAX_TOOL_WAIT_SECONDS` | — | exists |

Argued explicitly, since the brief asked: **model tier is not a ceiling.** A
child that needs a stronger model to do its narrow job should have it; what
must not widen is the *spend*, and the lease bounds that regardless of tier.
Making tier monotone would push callers to over-provision the root so
children can be strong — which raises spend, the thing we are trying to
bound. The subscription-tier constraint on *which* models exist is tenancy
and is already identical down the tree.

Second argument: **the expert's own weekly budget stays orthogonal** rather
than being folded into the lease. It is a per-identity churn guardrail
("this expert may not burn more than X/week on schedules"); the lease is a
per-tree spend bound. Two different questions; folding them would make a
delegation from a rich parent drain a frugal expert's week.

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

Why no sibling or peer edges: each additional edge is a second path for
authority and taint to travel, and it is a path that has to be bounded
separately. The tree carries every legitimate message with at most one extra
hop (through the parent), and the parent is precisely the node that holds the
envelope those messages must respect. A design with lateral edges has to
answer "which of the two parents' envelopes applies?" — and that question
has no monotone answer.

`handoff` is the deliberate exception: it re-roots. After a handoff the
receiving node's parent is the human; its questions reach Home; its budget
lease is *converted* into a root lease (the parent's remaining lease at the
moment of handoff is what it carries — nothing is created). Ownership moves;
authority does not widen.

What the roster (`<team_context>`) is for under this model: **addressing,
not context.** It tells a node which identities it may spawn a child *as*. It
is not a channel and it does not make those experts reachable — a listed
expert has no session until the caller spawns one.

### 3.4 Scale invariance — 3 agents and 3000

**Three.** A founder with AutoPilot, an Ops expert and a Growth expert.

```
human ── root turn (AutoPilot, interactive, lease = min($2, user remaining))
           ├── delegate → Ops    envelope{depth 1, children 0, lease $0.60, deadline +5m, tools ∩}
           │                     brief{task, authority: "no refunds, no dates", acceptance: "a reply draft"}
           │                     report{done, artefact: draft.md, spent $0.11}
           └── delegate → Growth envelope{depth 1, children 2, lease $0.80, deadline +10m}
                                 └── isolate (Growth) envelope{depth 2, children 0, lease $0.30}
```

Every tool the user sees is the tool that exists today. What changed is that
Ops *cannot* delegate onward (leaf), *cannot* exceed $0.60, *cannot* run past
the parent's deadline, and *knows* it may not promise a refund because the
brief said so. The ToolChain card shows the envelope next to the delegate
row. Total spend ≤ $2 by construction; the parent's remaining lease after
both spawns is $0.60, which is what its *own* remaining turn may cost.

**Three thousand.** An organisation of 400 users, 3000 hired experts, a few
hundred schedules firing across the day, plus graphs with `AutoPilotBlock`s.

- There is no global object. There is a forest; each tree is rooted in a
  turn; each turn's tree is bounded by `MAX_DEPTH = 3` and per-node fan-out
  (default 0 — leaves — so the *default* tree is a star of at most
  `children_remaining` nodes under the root; a parent that wants a deeper
  tree grants fan-out explicitly and pays for it out of its lease).
- Spawn cost is O(1) DB rows (the parent's, CAS) plus O(1) session reads.
  The existing O(depth) provenance walk goes away because depth is a field.
- Spend: every root lease is carved from the user's remaining daily/weekly
  cap (exists); every child lease is carved from its parent; therefore
  `Σ spend over the org ≤ Σ over users of their caps`, with no component
  ever summing anything. Conservation, not accounting.
- Concurrency: per-user running/in-flight caps (exist). 3000 experts is
  3000 *rows*; the number of live sessions is bounded by users × cap.
- Addressing: the roster injection is the one O(N) thing and it breaks
  first (3000 experts is ~150k tokens of roster in *every* first message).
  Replacement: `<team_context>` lists at most the caller's **pod** (or the
  first K by recent use when unpodded), and `list_team` gains a query so an
  expert is *found*, not enumerated. `resolve_target_expert` already accepts
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

What is *not* scale-invariant and I am saying so: the SDK's `max_budget_usd`
has a floor (`_MAX_BUDGET_USD_FLOOR`) below which it will not go, so the
smallest fundable lease is that floor and a very deep tree of very small
leases is refused rather than run. That is the right failure — a spawn you
cannot fund does not happen — but it means the design's granularity is
bounded below by the SDK, not by us.

---

## 4. The invariants, and where each is enforced

| # | invariant (from the brief) | holds by | enforcement seam |
|---|---|---|---|
| 1 | Authority never widens on descent; cannot be obtained from a third party | `tools = parent ∩ request`; the only creator of child sessions reads the **parent's persisted** envelope; there is no tool that grants permissions; `expert_admin` is denied to every expert session (exists) | `derive_child_envelope` + `create_chat_session`; `apply_tool_permissions` in the executor; `execute_tool` re-checks groups |
| 2 | Budget strictly sub-allocated; Σ subtree ≤ root | lease carved by CAS from the parent's row at spawn; child's SDK `max_budget_usd` clamped to its lease; unfundable spawn refused | `derive_child_envelope`; `_resolve_dynamic_max_budget_usd` gains a `min(…, lease)` |
| 3 | Taint propagates down and never launders | `tainted = parent.tainted or born`; every child is `origin="automation"`; a new session id no longer means a clean bit | `derive_child_envelope`; T9's `is_tainted()` reads `envelope.tainted` as a source |
| 4 | Locality | every bound is a field on the child or a CAS on the parent; no fan-in, no traversal, no global set; roster scoped to pod | by construction; roster change in `_team_context` |

**A note on invariant 3 as it stands on `dev`:** nothing on `dev` *sets*
taint, because T9 is a branch. This plan builds the carrier and the
propagation and one birth source (chat-platform sessions, `source_platform`
set — T9 §5.2), so that when T9 lands the laundering path is already closed
and its gate has one more boolean to read. The claim I can make on `dev`
alone is "taint, once set, cannot be laundered through a spawn"; the claim
"taint is set when it should be" is T9's.

---

## 5. Where T10 and T9 slot in

**T10's `consult_teammate`** is the `consult` edge: an envelope with
`children_remaining = 0`, `tools = ∅`, a lease of one bounded completion, no
session. It needs nothing from this design except the `authority` slot,
which it already discovered it needs. The pod reviewer becomes "the expert
the pod's members should consult before committing", exactly as T10 had it.

**T9's gate** reads `envelope.tainted` as a taint source alongside its
transcript scan, and — this is the part that matters at scale — its
"delegation is ALWAYS_ASK" rule can be relaxed to "delegation is ALWAYS_ASK
*when the child would be granted anything the gate would ask about*". A
leaf child with a $0.50 lease, no effectful tools, and a 2-minute deadline
does not need a human to approve it; the envelope is the approval. That is
the ergonomic win the two designs get by composing: T9's gate stops asking
about spawns the envelope has already bounded.

Neither integration is built here. Both are one-boolean / one-field seams.

---

## 6. Failure modes

| failure | behaviour |
|---|---|
| Parent crashes after carving a lease, before the child starts | Lease is gone from the parent, unspent by the child. **Conservative** — the tree under-spends. A sweep on `deadline_at` returns leases of children that never ran (v2); v1 accepts the leak because it errs toward spending less. |
| Child crashes mid-turn | Existing: stream error → `failed`. The lease was an upper bound; actual spend is recorded by the existing usage path. |
| Parent fans out past the per-user running cap | Existing: `rejected_concurrent_turn_cap` — the spawn fails, the lease CAS is rolled back (spawn is: carve → schedule → on failure, return). Not a deadlock: children are *rejected*, never queued behind a parent that is waiting on them. |
| Two parallel spawns race on the parent's row | CAS on `children_remaining` and `remaining_budget` — the loser re-reads and either fits or is refused. |
| Deadline passes while the child is running | Executor's deadline watcher fires `enqueue_cancel_task`; child finalises `failed`; report status `out_of_time`. |
| Child asks a question | `needs_input` up the tree; the root turns it into `ask_question`. A child in an automation tree with no interactive root (a schedule) lands one Home item at the root. |
| Legacy session with no envelope | Reads as `depth=0, children=default, lease=root default, tainted=False, tools=None`. A legacy *child* (has `delegated_by_session_id`, no envelope) reads as `depth=1`. Fail-open on depth for legacy rows is acceptable because `MAX_DEPTH` still applies from that row down. |
| Redis brown-out | Nothing here lives in Redis. Envelope is on the session row. |

---

## 7. What I am NOT building

- No orchestrator, planner, scheduler, router, or manager expert.
- No message bus, task board, work-item model, DAG, or queue beyond the one
  that exists.
- No lateral edges: no sibling, peer, or broadcast communication.
- No shared memory, shared session, or team-wide context. Memory scoping is
  untouched.
- No voting, debate, consensus, or multi-round negotiation.
- No lease *return* (unspent budget flowing back up) in v1 — it is the
  conservative direction and can wait.
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
2. **Budget is a stop, not a reservation, at the SDK boundary.** The CLI
   halts a turn once `max_budget_usd` is exceeded; it does not pre-reserve.
   Overshoot per node is bounded by one model call. Graph runs a child
   starts with `run_agent` are charged in credits to the user's wallet and
   the expert's weekly budget — **not** to the lease — in v1. Naming it
   rather than hiding it: the lease bounds LLM spend; the wallet and weekly
   budget bound graph spend; the two are not yet one number.
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

The smallest thing that demonstrates the core claim — *invariants hold by
construction, locally, at spawn* — is the envelope and its derivation, wired
into the three spawn paths:

1. `copilot/envelope.py` — `Envelope`, `SpawnRequest`, `derive_child_envelope`,
   `root_envelope`, `EnvelopeRefusal`. Pure. Property-tested in
   `envelope_test.py` (monotonicity of every field, refusal on every
   overflow, idempotent derivation).
2. `ChatSessionMetadata.envelope: Envelope | None` (no migration).
3. `run_sub_session`, `delegate_to_expert`, `handoff_to_expert` derive the
   child envelope from the parent session, refuse on `EnvelopeRefusal`, and
   pass the intersected tool set as `permissions`. `child_session_origin`
   becomes `"automation"` unconditionally.
4. `_resolve_dynamic_max_budget_usd` takes the session and clamps to the
   lease.
5. `chain_refusal` keeps its loop check, drops its depth walk in favour of
   `envelope.depth`.

Not in the slice: deadline cancellation, fan-out CAS (v1 uses a per-turn
in-memory count — the roast will have an opinion), lease return, roster
scoping, typed report. Each is a named follow-up.

---

## 10. Sensitive findings — where they are described

Two of §1.3's grounding facts are logged in `~/code/agpt/.claude/log/findings.jsonl`
as security-sensitive (permissions equal on descent with no narrowing path;
depth bound not counting same-identity sub-sessions). They are described in
this file only as facts and design consequences — §1.3 (two bullets), §3.2
(the `tools` and `depth` rows), §4 (invariants 1 and 3). No worked sequence
appears anywhere in this repository. Disclosure routing is Reinier's call;
this branch is not pushed.

## 11. The roasts

Two sub-agents, in parallel, different jobs. Both were given this file and
the code and told to break it, not improve it. Recorded verbatim in summary
below, with what changed and what I rejected.

### 11.1 Architecture roast

*(pending)*

### 11.2 Value roast

*(pending)*
