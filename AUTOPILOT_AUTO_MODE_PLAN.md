# AutoPilot auto mode — design

Auto mode is a permission gate in front of every AutoPilot tool call. It exists
so AutoPilot can run a long stretch of work without stopping to ask about
routine steps, while the actions that can cost real money, reach outside the
platform, or destroy data still require a person.

## 1. What problem this actually solves

AutoPilot does not prompt before acting today. Every tool in `TOOL_REGISTRY`
runs the moment the model names it. The gates that exist are narrow and
hand-placed:

| Gate | Where | Covers |
|---|---|---|
| Propose/confirm + user-turn watermark | `copilot/tools/expert_proposal.py` | `hire_expert`, `raise_expert`, `update_expert`, `update_expert_soul` |
| Human review of sensitive blocks | `copilot/tools/helpers.py::check_hitl_review` | `run_block`, for blocks flagged `is_sensitive_action=True` |
| Static deny + path scoping | `copilot/sdk/security_hooks.py` | SDK built-ins |
| Capability hiding | `permissions.py`, `tool_names_in_groups` | Whole tool groups on/off |

Nothing covers `bash_exec`, `post_to_chat_platform`, `write_workspace_file`,
`delete_workspace_file`, `delete_schedule`, `setup_agent_webhook_trigger`,
`run_mcp_tool`, `run_agent`, `browser_act`, or `edit_agent`, and nothing
anywhere knows whether the content driving those calls came from a web page.

So this ships the gate. The autonomy payoff follows from it: once a gate
exists, AutoPilot can be told to stop asking for permission in prose (§7),
which is what costs a long run its momentum today.

## 2. The central claim

**The classifier buys ergonomics, not security. The static tiers are written to
be safe on the assumption that the classifier always says "allow".**

Everything below follows from that. An LLM judging whether an action is safe is
useful for keeping a long run from stopping on every file write, and it is the
wrong thing to depend on when the arguments it reads may have been written by
whoever controls the web page AutoPilot just fetched. So the tiers carry the
security, and the classifier only removes friction inside the envelope the
tiers already make safe.

## 3. Where the gate sits

**Seam 1 — `copilot/tools/base.py::BaseTool.execute`.** Both engines funnel
every registry tool through this one method: baseline via
`tools/__init__.py::execute_tool`, SDK via
`sdk/tool_adapter.py::_execute_tool_sync`. One insertion point covers all 71
tools on both engines, including tools added later — there is no registration
list to forget to update.

**Seam 2 — `sdk/tool_adapter.py`, for non-registry tools only.** The file
handlers from `sdk/e2b_file_tools.py` (`Write`, `Edit`, `read_file`, and the
`E2B_FILE_TOOLS` entries) are registered directly onto the MCP server. They are
not `BaseTool` subclasses and are not in `TOOL_REGISTRY`, so seam 1 never sees
them. They sandbox their own paths, but a permission gate that skipped the main
file path in SDK mode would be a gate in name only. One condition —
`tool_name not in TOOL_REGISTRY` — catches them without double-gating anything
else.

SDK built-ins (`Read`, `Glob`, `Grep`, `WebSearch`, `Task`) are out of scope:
they are constrained by `security_hooks.py` and are not effectful outside the
sandbox (§10).

## 4. The decision

`gate/__init__.py::check_action` returns `ALLOW`, `ASK(reason)`, or `DEFER`.
Cheapest and most certain first; the classifier last and least trusted:

1. **Gate inactive** → allow. Today's behaviour, byte for byte.
2. **An approval for exactly these arguments** → allow, consumed single-use.
3. **Tier `DEFER`** → allow; a different, existing gate owns this call.
4. **Escalated this session** → ask.
5. **Tier `ALWAYS_ASK`** → ask.
6. **Session tainted and the tool is effectful** → ask, classifier skipped.
7. **Tier `READ`** → allow.
8. **Tier `JUDGED`** → classifier; anything but a clean allow → ask.

An unlisted tool is `JUDGED`, so forgetting to tier a new tool costs friction,
never silence.

### Why delegation is `ALWAYS_ASK`

`delegate_to_expert`, `handoff_to_expert` and `run_sub_session` each open a
**new session**, which starts with a clean slate for §5's provenance tracking.
`child_session_origin` returns `parent.origin or "automation"`, so an
interactive parent produces an interactive child. Delegation is therefore the
last point at which a person can see the work before it continues somewhere the
gate is not watching, which makes it the right place to ask.

### Why `run_mcp_tool` is `ALWAYS_ASK`

Its behaviour lives on a remote server named by `server_url`, and
`auto_lookup_mcp_credential` attaches the user's OAuth credential to the call.
There is nothing local to judge.

### Why `run_block` / `run_agent` / `continue_run_block` are `DEFER`

Their arguments are opaque identifiers — a preset UUID, a block UUID, a review
id. Asking a model whether a UUID is safe produces a verdict with no
information behind it, and the approval card would show a person the same UUID.
`check_hitl_review` already gates these against the *resolved* block, its
inputs and its credentials. The auto-mode gate stands down rather than layering
a blind judgement on top of a sighted one.

Generalised: **a tool whose arguments do not describe its effect is never
classified.** It either has its own gate, or it is always-ask.

### The escalation channel

A monotone, session-scoped set. Anything in it forces ask for that tool, and
there is no de-escalate. A rejection writes into it, so re-proposing the same
action with the arguments nudged cannot buy a fresh verdict. It is also the
seam a third-party veto — an authority review, a policy check — can write into
without either side importing the other.

## 5. Untrusted content

An autonomous agent acting on content it fetched is where injected instructions
turn into real actions, and the classifier is the obvious thing to attack: the
injected text would be sitting in the arguments it reads.

**So in that case the gate does not ask the classifier.** `gate/taint.py`
records, structurally, that a session has ingested bytes we did not author —
decided by our own control flow, not by an opinion about whether the content
looked dangerous. An attacker controls the phrasing; they do not control
whether `web_fetch` ran. Once a session is marked, the high-consequence writes
(`bash_exec`, `write_workspace_file`, `store_skill`, `memory_store`,
`schedule_followup`, the file handlers, …) go to a person instead.

Provenance is derived from the session transcript, with a Redis flag as a
same-turn cache for calls not yet written to history. That is durable by
construction, and it is set *before* a source runs rather than after it
succeeds — every MCP tool is annotated `readOnlyHint=True` so the CLI
dispatches calls in parallel, and a flag written on success would be readable
too late by a sibling call in the same batch.

Reading stays free after a session is marked. `web_fetch`, `web_search` and
`browser_navigate` are deliberately not escalated: escalating them turns every
research turn into an approval prompt, which is the nagging this feature exists
to remove. **A URL can still carry data outward, and this design does not stop
that** — the honest fix is egress control, not a smarter judge (§10).

Memory and skills are covered on both sides: the writers escalate when a
session is marked, and the readers (`memory_search`, `read_skill`,
`memory_forget_search`) are themselves provenance sources, because content
stored in one session is replayed into the next.

Sessions whose *prompt* is untrusted are marked at birth: a chat-platform
session's turns are authored through a linked server rather than by the account
owner typing, so they are not treated as first-party intent.

### On a dedicated injection detector

Perplexity and others have published models for detecting prompt injection.
The rule this design commits to, whether or not one is ever added:

> An injection detector may only ever escalate a decision toward ASK. It may
> never be a reason to allow something structural provenance would have
> stopped.

A detector's useful jobs are telling the approving person "this page appears to
contain instructions aimed at an agent" and escalating an otherwise-allowed
read. Both additive, neither load-bearing, neither in v1.

## 6. Scope and the on switch

- **Tiers are code constants** in `gate/policy.py`. Changing what is always-ask
  is a code review. No policy DSL, no admin UI, no plugin system.
- **The gate is active only in interactive sessions.** Automation sessions (the
  scheduler, `AutoPilotBlock`, sub-sessions) and legacy rows with no origin keep
  today's behaviour exactly, so **nothing already shipped changes**. This is
  deliberate, not an omission: parking a question in a run nobody is watching is
  a stall, not a safeguard, and refusing instead would break shipped behaviour —
  scheduled `post_to_chat_platform` ("post an update in #standup every Monday")
  fires from an `origin="automation"` session and would fail forever.

  > The gate asks at the moment a person is present to authorize. It never
  > parks a question where nobody is watching.

  Unattended work is authorized by the interactive act that created it, which
  is why delegation is always-ask and `schedule_followup` escalates.
- **`Flag.COPILOT_AUTO_MODE`**, default `False`. A rollout control, not a safety
  control — flag-off is today's behaviour, which is why it defaults off rather
  than fail-closed. Every failure *inside* an enabled gate fails toward ask.
- **`ChatSessionMetadata.auto_mode`** is a per-session override; `None` follows
  the flag. No new API surface in v1.
- **Two config knobs**: `gate_model`, `gate_timeout_s`. `gate_model` is
  deliberately independent of `title_model`, which
  `ChatConfig._apply_local_aux_models` rewrites on local deployments — silently
  swapping the model behind a permission decision is exactly the substitution
  nobody would notice.

## 7. What people see

An approval card in the chat naming the tool, showing its arguments and a
one-line reason, with Approve / Reject. This reuses the `PendingHumanReview`
rails that `run_block` already uses, so the card, Home's "Needs You" row, the
awaiting-review alert and the approve/reject endpoint all already exist and are
wired to a key the gate can write.

At most **one** pending gate approval per session: the model should stop at the
first gate rather than queue several, and a queue would otherwise produce one
alert per call.

Approvals are bound to a hash of `(session, user, tool, arguments)`, so an
approval means "you may do *this*", not "you may use this tool". They are
consumed single-use, with the delete's row count as the mutex, because parallel
dispatch means two identical calls can both observe an approval.

The model is told, only when the gate is active:

> Auto mode is on. Act — do not ask for permission in prose for reversible,
> in-scope steps; a gate will stop you when it matters. If a tool returns
> `approval_required`, say plainly what you wanted to do and why, then stop.
> Never retry it, never work around it, and never use a different tool to
> achieve the same effect.

## 8. Failure modes

| Failure | Behaviour |
|---|---|
| Classifier errors, times out, or returns an unusable body | ask |
| Redis unavailable | provenance falls back to the transcript, which is authoritative |
| DB unavailable (no approval row can be written) | refuse; never allow |
| Feature flag lookup fails | defaults `False` → today's behaviour |
| The gate itself raises | refuse. A gate that crashes must not become a gate that passes |
| An approval is already pending | refuse; do not queue |
| Approval race | the delete's row count decides |
| Unattended session | gate inactive; authorization happened interactively |
| Unknown tool | judged |

## 9. Files

**New** — `copilot/gate/`: `policy.py` (tiers, provenance sources, the
classifier rubric), `taint.py`, `review.py`, `classifier.py`, `mcp_seam.py`,
`__init__.py`, and four test modules.

**Changed** — `copilot/tools/base.py` (seam 1), `copilot/sdk/tool_adapter.py`
(seam 2), `copilot/tools/models.py` (`ApprovalRequiredResponse`),
`copilot/model.py`, `copilot/config.py`, `util/feature_flag.py`,
`copilot/prompting.py`, both engine services, and two small frontend touches (a
label for the new response type, and a correction to the post-approval message,
which previously assumed every approval was a `run_block` review).

## 10. Known limits

- **A URL can carry data outward after a session is marked** (§5). Not stopped
  by design; needs egress control.
- Approving from Home does not resume the chat. The resume step is triggered in
  the chat view, so an approval granted elsewhere leaves the session idle. The
  deep link exists (`_enrich_pending_reviews` sets `session_id`) but is not
  wired.
- SDK built-ins are not gated; their MCP file-tool replacements are.
- Auto mode is fixed at session creation, not toggleable mid-session.
- Provenance is one flag per session, not per content item. Coarse on purpose.
- No injection detector (§5).
- The gate is inactive in automation sessions by design (§6); their safety
  rests on the interactive act that authorized them.
