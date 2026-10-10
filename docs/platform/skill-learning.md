# Nightly skill learning

AutoPilot and Experts can turn work they actually verified into reusable
skills. This page describes what the first release does, what it does not
do, and how to inspect and control it. The feature is off by default and
enabled per cohort with the `dream-skill-learning-enabled` flag. The dreaming
pipeline runs learning after its memory stages when this flag is enabled.
Learning also has an independent nightly job, so it can run without enabling
other memory stages. Both paths share the same lease and revision ledger.

## What happens

1. **During work** the chat services record a bounded reference to each
   completed user turn as a *learning source revision*: message references,
   typed tool outcomes (an exit code, a completed run, a passing validation),
   and the user's explicit, unqualified confirmations. The record includes
   the persisted user message that started the turn, so a confirmation is
   attributed to a real stored message and never to an invented reference.
   No model writes a lesson at this point, and recording never blocks a
   reply.
2. **Overnight** (03:00 in the user's timezone, with a capped catch-up after
   downtime) a per-user job reviews eligible revisions. A source is eligible
   only when it still exists, still belongs to the same owner and Expert, is
   not excluded, learning is not paused for that Expert, and it carries a
   concrete outcome signal. Plans, unanswered questions, and unsupported
   "done" claims are skipped without a model call. Idle accounts cost
   nothing: with no new eligible work the pass exits before any model call.
3. The reviewer model (routed and billed like ordinary work) looks for
   work worth keeping: repeatable multi-step workflows, non-obvious error
   recoveries, and user corrections that improve recurring work. Trivial
   successes, generic advice, chat logs, and duplicate procedures are skipped.
   It proposes one
   bounded change: update an existing skill, create a meaningfully distinct
   one, or skip. Deterministic checks then require a valid slug, the required
   sections (`Steps`, `Verification`), a concrete verification statement, and
   citations of evidence spans that carry a checked outcome. Rejected
   metadata is never persisted.
   An existing skill can be rewritten only when its complete body was
   included in the review. Skills outside the bounded body window still
   appear by name and description to prevent duplicate creation.
4. **Publication** re-checks everything live inside one database
   transaction: source eligibility, epoch and revision, the skill's own
   policy, and the head version (compare-and-swap). A concurrent human edit,
   pause, or exclusion produces a conflict or a refusal, never an overwrite.
   The workspace `SKILL.md` is written afterwards, inside the registry's
   per-owner write lock and only while the head still names that version:
   a write resumed after a newer human correction is abandoned rather than
   placed on top of it. An interrupted write is finished (or abandoned) by
   the next run without a second model call.
5. **Retrieval** pins the exact version a conversation loads. A load is
   recorded as a load, never as a success. An existing conversation sees an
   overnight update at its next user turn without starting a new chat.

Every write to a skill — overnight, during work, an owner edit, a restore,
or an import — appends an immutable version with its origin and the
contributing source revisions. A version restored from history inherits
its base version's sources.

Learned skills stay private to the account and Expert that did the work.
Personal conversations teach the account's personal assistant; an Expert's
conversation teaches only that Expert. The learning process never creates
a marketplace listing or automatically shares a skill. Existing imported
skills require the owner's decision before an automated change is applied.

## Evidence and states

Evidence labels are concrete: **Worked once in the source**, **Requested by
user**, **Loaded N times · Outcome unknown**, **Reported as working by you**.
"Passed checks in N later uses" is reserved for recorded checks and is not
produced by an owner's report.

| State                    | Meaning                                                      |
| ------------------------ | ------------------------------------------------------------ |
| Ready to use             | Published and passed the applicable checks                   |
| Needs your decision      | A proposal for a human-controlled skill or an uncertain match |
| Blocked by content check | A credential-like value was found; only the pattern class and an ordinal step are recorded |
| Paused (learning)        | Future automated changes are off for the skill or Expert     |
| Paused (not in use)      | The owner stopped using the skill; history is kept           |
| Archived                 | Superseded, stale, or invalidated by a revoked source        |

## Controls

Learned skills can contain a concise `SKILL.md` entrypoint together with
reusable scripts, reference material, fixtures, and templates. The reviewer
uses code and material visible in verified evidence, and records the limits
of what was checked. A simple procedure can remain a single Markdown file.
Supporting files are saved with the immutable version, so retrying a save,
applying an update, or restoring a version preserves the corresponding
package. Older versions created before file snapshots were introduced
retain their existing Markdown-only restore behavior.

The Changes and decision views show added, changed, and removed package
files with bounded text previews. The instructions editor changes the
entrypoint; supporting file changes shown in a proposal are retained when
applying an edited alternative. New scripts are not executed automatically
during dreaming. A later task loads the package into its working directory
and runs the applicable checks.

Discovery triggers may contain up to 512 characters each. The per-turn
index displays a shortened combined hint when necessary; the full phrases
remain in the saved skill. Metadata and package constraints are checked
before a version is committed. A temporary storage failure retries the saved
package without another review; a permanent validation failure is settled
instead of retried indefinitely.

- **Expert page → Skills**: a recent-change line, per-skill learning lines,
  and a detail sheet with Summary, Changes (before/after by step, with the
  raw Markdown diff on request), Sources, and History. The sheet offers
  *Automatic improvements*, *Pause learning*, *Stop using this skill*, *Edit
  skill* (applies immediately), *Report an outcome* (evidence for a later
  review), and *Restore* (a new version; future use only).
- **Settings → Memory**: a learning history filtered by scope, origin, and
  state, and one list of open decisions across Experts. Personal (AutoPilot)
  skills open the same detail sheet here, with the same controls, since they
  have no Expert page.
- **Chat**: a once-per-load "Skill loaded" chip naming the exact version and
  origin. It links to that exact skill and version: the Expert page for an
  Expert skill, Settings → Memory for a personal one. A conversation also
  offers "Exclude from learning".

Editing always starts from the current version (restore a historical one
first to build on it). The version the editor saw is submitted with the
edit and checked under the same write lock, so a change that landed while
the owner was typing — overnight or from another tab — makes the save a
conflict with the draft kept, never a silent overwrite of that change.
A human edit turns automatic improvements off for that skill unless the
editor keeps them on. A restore also turns them off and records a
suppression for the undone behaviour, so the same change is not reapplied
silently; a near-identical proposal becomes a labelled decision instead.
Excluding a source makes every version that depends on it unavailable,
restores the newest independent version when one exists, and otherwise
pauses the skill with an explanation. A source the viewer can no longer
open shows **Evidence not visible to you** without a title or link.

## Adapters

The learner never inspects an application's schema. A source adapter
implements `revalidate(source_id, revision, scope, approval_event_id)` and
`load_evidence(...)`; approval-aware kinds also expose an approval
checkpoint (event id, actor, approved revision). Publication is conditional
on that live answer; a static approved flag in a queued payload is not
enough. The first release ships the ordinary-chat adapter only.

## Operators

`GET /api/skill-learning/status` reports the backlog, the oldest pending
source, the last review and last applied change, reviews still retrying,
and the last 30 days of review cost. Every review disposition — applied,
skipped, no novel procedure, deferred, conflict, blocked by content check,
provider error, budget exhausted, stale eligibility, inaccessible evidence,
suppressed — is kept per source revision in the review ledger.

`POST /api/skill-learning/run` runs the same bounded learning pass immediately
for the authenticated account. It preserves the normal billing, scope,
content, and ownership checks. A second run with no new evidence makes no
model calls. Runs renew their lease between sources and check it again after
each model review; losing the lease leaves the source pending without
publishing the proposed change.

## Not in the first release

A morning digest, per-section edit protection, selective forgetting
cascades, automatic sharing between Experts, skill merges or renames,
age-based archiving, additional source adapters, and any automatic
execution of a newly written recipe.
