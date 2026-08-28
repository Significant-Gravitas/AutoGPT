# Multi-expert teams in the Platform (TODO T10)

**Status:** plan + draft implementation, branch `pwuts/multi-expert-teams`.
Design in §2, the adversarial roast and what it changed in §3, the measured
result in §4.

---

## 0. The finding, up front

Reinier said plainly that "a team of experts cannot be shown to do something a
single expert demonstrably cannot" is a legitimate result. This lands between
that and a full team model, so the headline goes first — and it got smaller,
not larger, as the evidence came in.

1. **The coordination half of "teams" is already built and already works.**
   `delegate_to_expert`, `handoff_to_expert`, `list_team`, `<team_context>`, a
   3-hop chain bound, a loop guard, per-expert budget/pause, per-expert memory
   namespaces and `ExpertPod` grouping all exist on `dev` today. An expert can
   already hand work to a teammate who runs under their own soul, memory and
   budget. Building a planner, a router, a task board, a message bus or a
   manager expert on top of that is adding a second coordination layer to a
   working one. **I am not building it, and I do not think it should be built.**

2. **Almost everything else "team" evokes is either a boundary violation or
   theatre.** Shared memory breaks the enforced v1 isolation. A shared
   workspace is a slower `delegate_to_expert`. Debate rounds, voting and
   consensus multiply cost and latency for an output a single strong model
   already produces, and the T5 experiment
   (`AutoGPT5/EXPERT_GENERATOR_FINDINGS.md`) is direct evidence that stacking
   more LLM judgement on top of LLM output does not reliably improve it.

3. **One thing N experts do that one expert structurally cannot: check the
   work.** A self-review runs inside the context that produced the error. Any
   second reader in a fresh context fixes that, and this is the whole of what a
   "team" buys here.

4. **But the second reader does not need to be a second personality, and
   should not be.** T5's only reliable instrument was a *scripted, soul-free*
   audit (97.9% self-agreement); its evidence that souls matter is evidence
   that giving an **actor** a soul swings behaviour from 0.00 to 1.00 on this
   dimension. Putting a persona in the judge's seat imports the largest known
   source of variance into the control. §3.1 is where this plan was wrong and
   got corrected.

5. **The load-bearing mechanism turned out not to be the reviewer at all.** It
   is forcing the drafter to write down *what each commitment rests on* before
   anyone reads it. §4 measures this: with that one field, even a naive "is
   anything wrong with this draft?" gets the battery right; without it, no
   framing can — because the authorised and unauthorised drafts are the same
   bytes.

So: **a team is not an org chart. A team is a check — and most of the check's
value is a discipline the drafter imposes on itself.** What the teammate adds
on top is (a) a fresh context, (b) their declared limits as the policy the
check runs against, and (c) a name against the ruling. That is real, it is
small, and it costs one cheap model call.

## 1. What actually exists today (grounding)

Read before designing. All paths relative to `autogpt_platform/backend/`.

### 1.1 The Expert

`schema.prisma:933` — `model Expert`. Roster templates have
`ownerUserId = null, isTemplate = true`; hiring instantiates an owned copy
(`@@unique([ownerUserId, sourceTemplateId])`).

The **Soul** is four columns: `identity`, `voicePreferences`, `boundaries`,
plus `name`/`role`. `backend/api/features/experts/models.py` adds two
`PROTECTED_SOUL_RULES` that a user cannot remove:

```python
AI_DISCLOSURE_RULE = "The expert discloses that it is AI when acting externally."
EXTERNAL_ACTION_APPROVAL_RULE = "External actions require approval."
```

Other per-expert state: `weeklyBudget` + `schedulesPausedAt` (a breach pauses
that expert's schedules and triggers, never chat; `ExpertPauseEvent` is the
reversible audit log), `Workflows` (`ExpertWorkflow` join to marketplace
listings / library agents, with an install-time schedule), `ChatSessions`,
`GraphExecutions`, `AgentPresets`, and `podId`.

### 1.2 Memory isolation (the hard constraint)

`backend/copilot/graphiti/client.py:83` — `derive_memory_group_id()`:

```python
if expert_id is None:
    return f"user_{user_id}"          # AutoPilot keeps the legacy namespace
scope_digest = hashlib.sha256(expert_id.encode()).hexdigest()
return f"expert_{scope_digest}"
```

Every Graphiti read and write (`context.py:60`, `ingest.py:385/492`,
`communities.py:309`) derives its namespace from the *session's* `expert_id`.
There is no cross-namespace read path. The prompt states it as a rule too
(`prompting.py`, memory supplement): *"Memory is private and isolated to the
current assistant. AutoPilot and hired experts cannot read each other's
memories."*

**This plan does not touch it.** See §4.

### 1.3 How an expert is invoked

- `build_expert_identity_suffix()` (`copilot/expert_context.py`) renders
  `<expert_identity>` — name, role, identity, fenced voice, boundaries,
  protected rules — and appends it to the **system prompt on every turn**, so
  Soul edits reach live sessions while the cacheable prefix stays
  byte-identical. Expert sessions **fail closed**: a missing/archived expert
  raises `ExpertSessionUnavailableError` rather than silently running as
  AutoPilot.
- `build_expert_context()` injects, into the **first user message only**,
  `<expert_workflows>` (installed workflows to prefer `run_agent` on) and
  `<team_context>` (the roster, self excluded).
- `fence_voice_preferences()` blockquotes user-authored voice text as style
  data, because the hire flow's paste-your-own path can carry externally
  sourced text into a system-priority sink.

### 1.4 Cross-expert work today

| tool | file | shape |
|---|---|---|
| `delegate_to_expert` | `copilot/tools/delegate_to_expert.py` | borrow a teammate: new `ChatSession` bound to the target, one full turn on `copilot_executor`, blocking up to `wait_for_result`, resumable, pollable via `get_sub_session_result` |
| `handoff_to_expert` | `copilot/tools/handoff_to_expert.py` | give the task away: same machinery, returns `status="transferred"`, caller cannot poll |
| `list_team` | `copilot/tools/list_team.py` | authoritative roster read (the `<team_context>` snapshot goes stale) |
| `run_sub_session` | `copilot/tools/run_sub_session.py` | same-scope context isolation, *not* a hand-off |

Shared policy lives in `copilot/tools/expert_delegation.py`:
`MAX_DELEGATION_DEPTH = 3`, a `seen`-set walk up `delegated_by_session_id`
that refuses handing work back to an expert already in the chain,
`safe_caller_name()` against forged preambles, and a roster-carrying
"unknown target" message.

Gating: `TOOL_GROUPS` puts these in `"delegation"` / `"experts"` /
`"expert_admin"`; `expert_tool_disabled_groups()` hides staffing tools from
expert sessions and expert-session tools from AutoPilot; `execute_tool()`
re-checks the group so hiding is an enforcement boundary, not a presentation
filter.

### 1.5 Pods

`schema.prisma:1045` — `model ExpertPod`: `userId`, `name`, `Experts[]`,
`@@unique([userId, name])`. Routes at `experts/routes.py` (`POST /experts/pods`,
`GET /experts/pods`, `PATCH /experts/{id}/pod`). The frontend groups the roster
by pod (`app/(platform)/team/helpers.ts::groupExpertsByPods`).

**A pod does nothing.** It is a visual grouping on one page. It does not affect
routing, context, budget, permissions or delegation.

### 1.6 AutoPilot's relationship to experts

AutoPilot is the engine, not a roster row: a plain session has
`expert_id = None`, uses the `user_<id>` memory namespace, and gets an empty
identity suffix (deliberately, to keep the system prompt byte-identical for
cross-user prompt caching). The `<expert_identity>` block tells the expert that
"the base instructions above describe AutoPilot, the platform engine you run
on" and that it always speaks as itself.

AutoPilot holds the `expert_admin` group (hire/raise/update); experts are
denied it — *an expert must not staff its own team*. AutoPilot can delegate to
any expert.

### 1.7 What T5 established about evaluating any of this

`AutoGPT5/EXPERT_GENERATOR_FINDINGS.md`, and it is load-bearing here:

- Holistic LLM rubrics scoring *soul text* were worthless: ρ = −0.08 with
  session quality, ρ = **0.88** with the *length* of the identity field, up to
  9 points of re-run noise across a 10.3-point spread.
- A **narrow bait audit** — one scenario, one checkable question, verdicts with
  verbatim quotes — agreed with itself **97.9%** of the time, matched a blind
  human read, and cleanly separated `0.00` from `1.00` compliance between
  siblings the rubric had put 5 points apart.
- Two of three baited scenarios produced *no* variance: Sonnet 5 refuses to
  invent statistics, forge quotes or claim a fake SOC 2 with or without a soul.
  **The base model already covers most of the honesty surface.**
- The exception — the only place a soul changed behaviour, stably across
  reruns — was **authority**: whether the expert commits the company to money,
  dates, refunds, discounts or policy the founder had not approved. Compliance
  ran from `0.00` (`operations__gen3`, `operations__human`) to `1.00`
  (`operations__gen1`) on identical inputs.
- And the sharpest failure in the whole corpus is a *within-context* one:
  `marketing__gen2` refused to invent a customer quote and then wrote one two
  paragraphs later, in the same reply; `marketing__human` refused a fabricated
  stat and then wrote a softer version of it. **Stating a rule and violating it
  three paragraphs later is exactly what the author of the text cannot see.**

Three consequences for this plan, and they shaped every decision in it:

1. Evaluate the team **behaviourally**, on a question with a right answer, not
   with a quality rubric. Anything else is unmeasurable.
2. Aim at **authority**, not honesty. Honesty is free from the base model;
   authority is the residue where the soul actually moves the output.
3. The failure a second reader catches is real and documented in our own data.

---


## 2. The design

### 2.1 What a team IS (data model)

**Nothing new.** No migration, no column, no join table.

The first draft of this plan gave `ExpertPod` a `reviewerExpertId` so a pod
could designate its checker. The roast killed it (§3.6) and it is gone. The
short version: enforcement would have been a prompt string either way; both
delivery channels for "your reviewer is X" are broken (`<team_context>` is
injected into the *first user message only*, and `get_delegation_supplement()`
must stay a byte-identical constant for cross-user prompt caching); plain
AutoPilot — the worst arm in T5 at 0.25 bait compliance — cannot be in a pod at
all; and the v2 gate the column was buying an option on does not exist.

So a team is the roster the platform already has, and `ExpertPod` stays what it
is: a folder on the `/team` page. The teams work is one verb, not one schema.

### 2.2 How work is routed between members

**Unchanged.** `delegate_to_expert` and `handoff_to_expert` already do it, with
a depth bound and a loop guard. No router, no planner, no capability index, no
auto-assignment. The model reads `<team_context>` or calls `list_team` and
picks.

### 2.3 The one new capability: `consult_teammate`

```
consult_teammate(expert_id, work, authority, question="")
  -> { verdict: "pass" | "block" | "insufficient",
       reason, quotes[], reviewer{id,name,role,avatar_url,color} }
```

**One structured LLM call. Not a session.** The audit frame is a fixed prompt
carrying the teammate's `boundaries` as fenced policy; the caller's `work` and
`authority` are the material. Mirrors `copilot/briefing/narrative.py`, which
already does soul-informed structured completion via
`copilot/dream/llm.py::structured_completion` with capped input, bounded
output, and `persist_and_record_usage` for the spend.

| | `delegate_to_expert` | `consult_teammate` |
|---|---|---|
| cost | a full agentic session | one call, **measured $0.0010** (§4) |
| latency | seconds to minutes, needs polling | one cheap-model call, inline |
| recursion | bounded at depth 3 by a session walk | **impossible: no tool surface** |
| output | prose the caller can rationalise away | parsed enum + verbatim quotes |
| purpose | a teammate does *work* | a teammate gives a *ruling* |

**The `authority` argument is the design.** The caller must enumerate every
commitment its draft makes and what each rests on — the user's own words, a
system confirmation, or nothing. Asking the reviewer to *find* the approval
instead would be unanswerable (§3.3) and would make "paste your whole context
in" the only way to pass. §4 shows this field is where the accuracy comes from.

**Bounds.** `MAX_WORK_CHARS = 6000`, `MAX_AUTHORITY_CHARS = 2000`,
`MAX_POLICY_CHARS = 1500`, `MAX_OUTPUT_TOKENS = 400`, 20s timeout,
`MAX_CONSULTS_PER_TURN = 3`. Model is `config.title_model`
(`anthropic/claude-haiku-4-5`), normalised for the active transport — *not* the
turn's own model, which is the one whose judgement is under test.

**Refusals**, reusing the existing helpers: never yourself, never an archived
teammate, and on a miss return the roster (`unknown_target_message`). A
*paused* teammate is allowed, unlike delegation: a consult starts nothing on
their side, and withholding a check from a team already under budget pressure
is backwards. Tool group `"delegation"`, so it rides the `HIRE_EXPERTS` flag
and works from plain and expert sessions alike.

### 2.4 The memory / visibility boundary

**Unchanged, and narrower than the first draft.** The reviewer gets its own
`boundaries`, the caller's `work`, and the caller's `authority`. Not the
caller's conversation, not the caller's memory, not its own memory, and — after
§3.1 — not its own `identity` or `voice_preferences` either. It writes nothing
to any memory namespace.

Two honest corrections the roast forced:

- **"Writes nothing" is false as literally stated.** `persist_and_record_usage`
  appends a `Usage` record to the calling session and writes a
  `PlatformCostLog` row. Neither is readable by another expert, so the
  isolation property holds; the sentence should say *writes nothing another
  expert can read*, and now does.
- **The verdict text becomes the caller's memory.** `reason` and `quotes` are
  generated under the reviewer's declared policy and land in the caller's
  transcript, which is ingestible into the *caller's* namespace. One-way, small,
  and disclosed rather than denied.

`insufficient` is a first-class verdict: unreadable input, a provider failure, a
timeout, or a dry-run session all land there, never on `pass`. A check that did
not happen must never read as one that did.

### 2.5 Who arbitrates

The reviewer arbitrates **by veto, on commitments only**: does the draft state,
promise or imply a commitment the authority list does not cover? Not taste, not
strategy, not tone. §4 measures that the scoped frame holds that line and the
unscoped one does not.

On `block` the caller must either remove the flagged lines or **say out loud to
the user that it is overriding the objection and why**. Silent override is the
one forbidden move. Escalation goes to the user — never to a third expert,
never to a vote, never to another round of argument. Two agents that escalate
to each other can oscillate; one that escalates to a human cannot.

**An unattended run has no human to escalate to.** Scheduled expert workflows
and `origin="automation"` sub-sessions have nobody watching, so there a block
means *do not send* rather than *ask someone* — the tool says so explicitly in
that case. This came out of the roast (§3.7).

### 2.6 What the user sees and controls

**Sees.** A `consult_teammate` row in the chat's ToolChain rendering the
reviewer's avatar, a verdict chip (No objection / Blocked / Not checked), the
reason, and the quoted lines — as plain text, never markdown, because that
string is model output conditioned on a user-editable Soul. The check is an
artefact the user reads, and the drafter's next message shows the fix or the
override in the same transcript.

**Controls.** Who they hire, and whether their `boundaries` say anything worth
checking against. There is no new setting, deliberately.

**The limitation, stated plainly and not buried:** "consult before committing"
is a *prompt rule*. It is advisory. A model confident enough to promise a refund
it should not is exactly the model that may not call the tool — so v1 catches
the careless case and misses the confident one, and the confident one is the
one that costs money. This is the crux, not a footnote.

The hard version needs an interception point, and one is being built: T9
(`AutoGPT3`, `pwuts/autopilot-auto-mode`) puts a 3-tier gate — READ /
ALWAYS_ASK / JUDGED, failing closed to "ask" — inside `BaseTool.execute()` and
reuses the `PendingHumanReview` rails. `consult_teammate` is `READ` in that
taxonomy: no effects outside the conversation, so it never needs approval
itself. A recorded `block` is a natural input to that gate's decision on the
*outbound* action. **The two must not become overlapping gates**: T9 owns
enforcement, this owns the domain judgement it lacks.

Note for whoever builds it: `EXTERNAL_ACTION_APPROVAL_RULE` is not that seam.
The roast checked, and it is defined in
`api/features/experts/models.py` and rendered into `<protected_rules>` by
`copilot/expert_context.py`. Nothing reads it.

### 2.7 What is NOT being built

- No orchestrator, planner, scheduler, or manager/lead expert.
- No shared memory, shared workspace, shared session, or team-wide context.
- No task/work-item model, no DAG, no board, no queue.
- No voting, consensus, debate rounds, or negotiation.
- No parallel fan-out or map-reduce across experts.
- No skill-matching or auto-routing engine.
- No pooled or team-level budget.
- No new chat surface or "team room".
- No change to `derive_memory_group_id` or any memory scoping.
- No schema change, no migration, no new route.
- No hard gate on external actions — that is T9's seam, named above.
- No new expert *kind*. Any teammate can be asked.

### 2.8 Files and call sites

| path | change |
|---|---|
| `backend/copilot/tools/consult_audit.py` | **new** — the audit frame, its bounds, the verdict model |
| `backend/copilot/tools/consult_teammate.py` | **new** — the tool |
| `backend/copilot/tools/models.py` | `ConsultVerdictResponse`, `ConsultingExpertInfo`, `TEAM_CONSULT` |
| `backend/copilot/tools/__init__.py` | registry entry + `TOOL_GROUPS[...] = "delegation"` |
| `backend/copilot/permissions.py` | `"consult_teammate"` in `ToolName` |
| `backend/copilot/context.py` | `MAX_CONSULTS_PER_TURN`, `take_consult_slot`, reset per turn |
| `backend/copilot/sdk/tool_adapter.py` | reset the same budget in the SDK engine's setter |
| `backend/copilot/prompting.py` | the check rule in `get_delegation_supplement()` |
| `backend/copilot/expert_context.py` | one line in the expert-session team rule |
| `frontend/.../ToolChain/ConsultCard.tsx` | **new** — the verdict card |
| `frontend/.../ToolChain/{toolCatalog.agent.ts,ToolResult.tsx}` | wire it in |

The per-turn budget is a `ContextVar` reset by both engines' `set_execution_context`,
so it is per turn *and* per asyncio task — concurrent turns in one worker do not
share it. Its limits are in §3.4.

---

## 3. The roast, and what it changed

A sub-agent was given the first draft of this plan, `EXPERT_GENERATOR_FINDINGS.md`
and the code, and told to break it — not improve it. Seven angles, its verdicts,
and my response. Three hits changed the design materially; one of them changed
it more than anything else in this document.

### 3.1 "Multi-agent theatre — the reviewer's soul is the wrong variable"

**LANDS. Accepted; the design changed.**

The first draft welded two variables together — *fresh context* and *different
soul* — and claimed the evidence supported both. It does not. What
`EXPERT_GENERATOR_FINDINGS.md` shows is that the one instrument in that corpus
that measured reliably (the bait audit, 97.9% self-agreement) was a **scripted,
soul-free judge**, and separately that giving an **actor** a soul swings this
same dimension from 0.00 to 1.00 compliance. Putting a persona in the judge's
seat therefore imports the largest known source of variance into the control.
The roast also caught me stretching a quote: §5.2's "a text reviewer cannot see"
means *the rubric reading the soul text*, not "the author of the text".

**Change:** the auditor is no longer a persona. `consult_audit.audit_frame()` is
a fixed prompt. `identity` and `voice_preferences` never reach it. The teammate
contributes `boundaries` as fenced policy and their name for accountability.

§4 then measured what the policy is actually worth: 3/18. Real, small, and the
only place in the run where *which teammate you ask* changed the answer.

### 3.2 "This should be a prompt change, not a feature"

**PARTIALLY LANDS. The zero-architecture alternative is refuted by our own data.**

"Just add a sentence to `boundaries`" is precisely `operations__gen4`, which
earned 5/5 on the boundaries dimension for the rule *"Never state or imply a
delivery date, refund, credit, discount, price, SLA, or policy exception the
founder has not explicitly given you"* — and then, in both runs, promised the
refund, comped the invoice and invented a fix date. Stating the rule in the
soul does not produce the behaviour. That is the strongest single argument for
doing anything at all here.

The roast is right that the licensed intervention is far smaller than the first
draft: one function and one prompt constant. That is now what this is.

### 3.3 "The design is internally contradictory" — *the strongest hit*

**LANDS. This broke the first draft and forced the central change.**

The original veto question asked whether a commitment was one *"the provided
context does not show the user authorising"*, while §2.4 deliberately withheld
the caller's conversation, the caller's memory, and the reviewer's own memory —
which is exactly where authorisation evidence lives. So the literally correct
answer for any commitment-bearing draft is always `block`, and the only route to
a correct `pass` is for the drafter to paste its conversation and memory dump
into `content`: the boundary erosion §2.4 exists to prevent. Correct, structural,
and I had not traced a `pass` case.

**Change:** the caller now supplies `authority` explicitly and the audit asks a
*closed* question — does `work` exceed `authority`? Nothing has to be found; the
mapping is checked. §4's `covered` row is the direct test: identical draft,
authorising context supplied, `pass` 3/3 on every arm that received it.

That change turned out to be the most valuable thing in the plan (§4).

### 3.4 "The per-turn cap leaks"

**LANDS, partially, and the residue is disclosed rather than fixed.**

Two of the roast's three mechanisms do not apply to what was built: the cap is a
`ContextVar` (not an instance attribute on a registry singleton, so no
cross-user bleed), and it is enforced inside `_execute`, which **both** engines
reach — the SDK path calls `BaseTool.execute` directly and bypasses
`execute_tool`, so a cap placed there would indeed have been a no-op.

The third is real and stands: **a delegated turn is a different session on a
different worker, so it gets a fresh budget.** With `MAX_DELEGATION_DEPTH = 3`
there are four sessions in a chain, so the true worst case is 12 consults per
user request — about **$0.012** at the measured price. And the roast is right
that `run_sub_session` does not write `delegated_by_session_id`, so the hop
counter re-zeroes through a sub; that is a pre-existing property of the
delegation bound, not something this tool introduces, but it does mean the
per-request ceiling is soft. Given the measured unit cost I am accepting that
rather than building a distributed counter for it.

### 3.5 "Memory isolation is violated in practice"

**PARTIALLY LANDS. Both corrections are in §2.4.**

The enforced boundary survives: `derive_memory_group_id` is untouched, there is
still no cross-namespace read, and the consult performs no Graphiti write. But
the first draft's "no shared context is introduced anywhere" was too strong on
two counts — `persist_and_record_usage` does write a session `Usage` record and
a cost-log row (neither readable by another expert), and the reviewer's ruling
lands in the caller's transcript and is ingestible into the *caller's*
namespace. Both are now stated rather than denied.

The fig-leaf charge — "the caller chooses what to send" is not a boundary if the
success path requires sending everything — was true of the first draft and is
what §3.3 fixed. `authority` is a narrow, purposeful field, not a context dump,
and §4 shows a two-sentence authority list is enough.

### 3.6 "The pod-reviewer column does not earn its migration"

**LANDS. Deleted.** See §2.1. The roast verified that
`EXTERNAL_ACTION_APPROVAL_RULE` — the "existing seam" the column was buying an
option on — is defined and rendered but read nowhere, and that both channels for
telling a model who its reviewer is are structurally unable to carry it
(`<team_context>` is first-message-only; the supplement must stay a constant for
prompt caching). Removing it took a migration, a route, four DB functions and a
schema change out of the change.

### 3.7 The rest

- **Unfenced verdict text.** Would have landed against the plan; does not land
  against the code. `_verdict_response` blockquotes the ruling with explicit
  provenance, exactly as `fence_voice_preferences` does, and the frontend card
  renders `reason`/`quotes` as plain text.
- **Fails open when the reviewer is gone.** Half-accepted: an archived reviewer
  is refused loudly with the roster; a *paused* one is now deliberately allowed,
  because a consult starts nothing on their side.
- **`dry_run` spends real money.** Accepted and fixed — a dry-run session returns
  `insufficient` without calling the model.
- **No user to escalate to in an unattended run.** Accepted and fixed — see
  §2.5. This was the best small catch in the roast.
- **Reviewer == delegator is permitted.** Accepted as harmless: a consult is
  stateless and one-shot, and the block travels back up the delegation result
  anyway.
- **The demo could not be run as specified.** Accepted entirely. The original
  demo compared a fresh treatment arm against T5 control numbers produced on a
  *tool-less* harness, had no negative control, and no arm that could show the
  feature was unnecessary. §4 is the rebuilt version: every arm run in one
  harness, three negative controls, and a control arm designed to beat me.

### 3.8 What the roast did not move

Its closing line was that the negative result in §0.1 is the real deliverable
and the rest is "an unrequested feature reasoning from a misread sentence". Half
right. The misread is real and §3.1 fixes it. But the residual claim does not
need that sentence: `operations__gen4` stating a rule and breaking it three
paragraphs later is a within-context failure, and no amount of "check your own
draft" reaches it, because the check runs in the context that produced it. §4
puts numbers on how much a fresh reader is worth. It is less than I first
claimed and more than nothing.

---

## 4. What it actually does — measured

Six fixed drafts, four arms, three runs each. 72 calls per run,
`anthropic/claude-haiku-4-5`, **$0.075** total, **$0.00104 per consult**. Run
twice independently; **the two runs agree in all 24 cells.** Harness kept local
(experiments do not ride the branch): `experiments/consult_teammate/`.

The drafts are held fixed on purpose — the auditor is the only new thing, so it
should be the only thing varying. `uncovered` and `covered` are the *same bytes*
with different authority lists, which is what makes the comparison sharp.

**Arms**, ordered from "could make this feature unnecessary" upward:

| arm | gets the draft | gets `authority` | frame |
|---|---|---|---|
| `draft_only` | yes | **no** | "is anything wrong with this draft?" |
| `unstructured` | yes | yes | "is anything wrong with this draft?" |
| `audit_no_policy` | yes | yes | the production audit frame, no teammate policy |
| `audit_with_policy` | yes | yes | the production frame + the teammate's `boundaries` |

| arm | correct | false blocks | missed |
|---|---|---|---|
| `draft_only` | 9/18 | 6/9 | 3/9 |
| `unstructured` | **15/18** | 3/9 | 0/9 |
| `audit_no_policy` | 12/18 | 6/9 | 0/9 |
| `audit_with_policy` | **15/18** | 3/9 | 0/9 |

And what each arm blocked *on* — classified from its own stated reason:

| arm | in-scope (a commitment) | out-of-scope (tone, spelling, blame) | unclear |
|---|---|---|---|
| `draft_only` | 5 | 3 | 4 |
| `unstructured` | 9 | **3** | 0 |
| `audit_no_policy` | 15 | **0** | 0 |
| `audit_with_policy` | 11 | **1** | 0 |

### 4.1 Four things this says

**1. `authority` is the load-bearing part of the design, not the reviewer.**
`draft_only` and `unstructured` run the *identical* naive prompt; the only
difference is whether the caller stated what its commitments rest on. 9/18 →
15/18. The `covered` row is the clean demonstration: the draft is byte-identical
to `uncovered`, so no reviewer of any quality can separate them without that
field — `draft_only` blocks the authorised draft 3/3, every arm that receives
the authority list passes it 3/3. **Most of the value here is a discipline the
drafter imposes on itself, and it needs no second agent at all.**

**2. The audit frame does not buy accuracy. It buys scope.** Same 15/18 as the
naive prompt — but the naive prompt spends 3 of its 12 blocks objecting to
tone, spelling and blame-shifting, once quoting the signature line "Frankie".
The scoped frame does that 0–1 times in 12–15. That difference does not show up
in a score and it is the one that matters in production: a check that objects to
your prose is a check the drafter learns to override, and then the real block
gets overridden too.

**3. The teammate's declared limits are worth 3/18 — the only place "which
teammate" changed anything.** `audit_no_policy` blocks a pure
escalate-to-the-founder reply 3/3 as an uncovered commitment. With the
teammate's `boundaries` ("Money and unhappy customers go to the founder, not out
the door") the same frame passes it 3/3, because escalation is now visibly the
policy rather than a promise. That is the measured content of "a *teammate*
checks it" over "a check happens".

**4. Nobody missed a real uncovered commitment except the arm with no authority.**
Every arm that got the authority list caught all 9, including `soft_date` —
"You'll hear from us with a resolution by end of day tomorrow", buried in an
otherwise blameless escalation. `draft_only` passed that one 3/3. Worth noting
that scenario exists because the *tool* caught it in the first smoke run and I
had mislabelled it: I wrote that draft as a negative control believing it
committed to nothing.

### 4.2 What this does not show

Stated plainly, because the whole point of §3 was not overclaiming.

1. **n is tiny.** Six scenarios, three runs, one domain (customer billing), one
   model. This is a smoke test with controls, not an eval.
2. **I wrote both the scenarios and the frame I am testing.** The `draft_only`
   and `unstructured` arms are the guard against that, and they are the reason
   claim 1 above is deflationary rather than flattering.
3. **`style_flaw` is contested and scored as a loss for everyone.** All four arms
   block it. The two audit arms block it for one defensible in-scope reason
   ("I'll look into it when I get a chance" as an uncovered service commitment);
   the two naive arms block it for typos and tone. Scored identically, not the
   same behaviour — which is exactly why the scope table exists.
4. **The drafts are fixed.** This measures the auditor, not the loop. Whether a
   model that receives a `block` actually fixes its draft, or overrides it in
   prose the user skims past, is not measured here and is the obvious next test.
5. **Nothing here tests an adversarial reviewer.** A teammate whose `boundaries`
   were poisoned via `update_expert_soul` is a real risk (§3.7); fencing is a
   mitigation, not a measurement.

### 4.3 Reproducing

```
cd experiments/consult_teammate
ANTHROPIC_API_KEY=... <backend venv>/bin/python run.py --runs 3
<backend venv>/bin/python analyse.py results.json
```

It imports `consult_audit` from the backend, so it exercises the production
prompt rather than a copy that drifts.

---

## 5. Does a team beat a single expert here?

Narrowly, on this evidence, yes — and less than the framing suggests.

**What a second expert genuinely adds:** a reader in a fresh context (catches the
`operations__gen4` failure that no self-review can, because self-review runs in
the context that produced it), that teammate's declared limits as the policy the
check runs against (measured: 3/18), scope discipline that keeps the objection on
commitments instead of prose (measured: 3/12 fewer out-of-scope blocks), and a
name on the ruling so the user can see who objected. Cost: **$0.001** and one
cheap-model call.

**What it does not add, and this is the honest half:** the largest measured
effect in §4 is not the teammate. It is making the drafter write down what its
commitments rest on before anyone reads them — worth 6/18, available with no
second agent, no team, and no new architecture.

**What "team" does not mean here at all:** coordination. The platform already has
delegation and hand-off with a depth bound and a loop guard, and they work. A
planner, a router, a task board, a manager expert or a debate protocol on top of
that would add cost, latency and failure modes for benefits nobody has
demonstrated. The right team feature was one verb and no schema.

If the wider question is how expert contexts should couple, what a sub-agent
inherits, and whether any of it survives 3 → 3000 agents, that is a different and
much larger piece of work. This tool should slot into it as a primitive; it is
not an answer to it.
