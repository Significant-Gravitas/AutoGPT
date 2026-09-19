# Run judge (TypeSafe Jev)

After every COMPLETED or FAILED run, the platform generates an "activity
status": a short prose summary of what the run did plus a `correctness_score`
from 0 to 1. Both used to come from one free-form chat-model call
(`backend/executor/activity_status_generator.py`).

The **run judge** (`backend/executor/run_judge.py`) replaces the numeric half
with typed judgments from [Jev](https://docs.typesafe.ai), TypeSafe's System
One model. One stateless request carries the full execution evidence as
`state` and asks six named questions. Jev returns, per question, a choice with
a probability for every option and a confidence (the top-two margin). Jev never
writes free text; the prose summary still comes from the existing chat model,
optionally conditioned on the verdicts.

It sits behind the same LaunchDarkly flag as the activity status
(`ai-agent-execution-summary`, `Flag.AI_ACTIVITY_STATUS`).

## Questions

All six are asked in a single request (the Jev AskMany shape). Option keys
are fixed and are what gets persisted.

| Key | Type | Options / levels |
| --- | --- | --- |
| `delivered` | Choice | `delivered`, `partially_delivered`, `not_delivered`, `cannot_tell_from_evidence` |
| `errors_vs_outcome` | Choice | `no_errors`, `errors_recovered_outcome_unaffected`, `errors_degraded_outcome`, `errors_caused_failure` |
| `failure_cause` | Choice (always asked; "if delivered, answer not_applicable") | `not_applicable`, `bad_user_input`, `missing_credential_or_integration`, `external_service_failure`, `agent_design_or_wiring`, `platform_bug` |
| `user_action_needed` | Choice | `none`, `fix_the_input`, `connect_an_integration`, `add_credits`, `rebuild_the_agent` |
| `external_side_effects` | Choice | `none`, `intended_only`, `unintended_or_repeated` |
| `output_quality` | Score, 5 levels lowest-first | `unusable`, `poor`, `acceptable`, `good`, `excellent` |

The wording lives in `backend/executor/run_judge_questions.py`. Every option
names the JSON fields of the evidence it should be judged from
(`graph_info.description`, `nodes[].recent_errors`, `is_graph_output`,
`overall_status.graph_error`, ...). `output_quality` is only meaningful when
outputs are visible; the instruction tells Jev to pick the lowest honest level
with low confidence otherwise and to record "not visible" via
`delivered=cannot_tell_from_evidence`.

### Evidence

The state is the same execution summary the prose judge receives, with one
improvement: terminal nodes (no outgoing link) and `AgentOutputBlock` nodes
are marked `is_graph_output: true` and keep up to 2,000 characters of output
per sample instead of 100. The whole request is bounded by the TypeSafe
client's 30,976-byte budget; oversized state is truncated with a note that is
persisted alongside the verdicts.

### Deterministic short-circuits

Runs that FAILED with a structured `failure_reason` of `insufficient_balance`
or `entitlement_required` never call Jev. They get a fixed record with
`source: "deterministic"`, `delivered=not_delivered`,
`user_action_needed=add_credits`, and a derived score of 0.

## Persisted record

`GraphExecutionStats.judge` (JSON in `AgentGraphExecution.stats`) holds:

- `answers`: every question's `choice`/`score`, `probabilities`, `confidence`
- `derived_correctness_score` = `P(delivered) + 0.5 * P(partially_delivered)`
- `request_id`, `latency_ms`, `input_tokens`, `output_tokens`
- `request` and `response`: the verbatim HTTP bodies (transparency contract)
- `truncated`, `truncation_note`, `error`, `source`, `mode`, `version`

`without_activity_features()` scrubs `judge` together with `activity_status`
and `correctness_score` when the flag is off for the viewer. A tokens-only
`PlatformCostLog` row (`block_name="run_judge"`, `provider="typesafe"`) is
written per Jev call; TypeSafe reports no price.

## Settings

| Setting | Values | Default | Notes |
| --- | --- | --- | --- |
| `TYPESAFE_API_KEY` | secret | empty | Platform-side key. Empty disables the judge (logged at debug). |
| `RUN_JUDGE_MODE` | `off`, `shadow`, `primary` | `shadow` | See below. |
| `RUN_JUDGE_TIMEOUT_SECONDS` | 0 < t <= 60 | `10` | Hard bound on what the judge can add to run finalization. |

### Modes

- **off**: Jev is never called; behaviour is exactly the pre-judge platform.
- **shadow**: the existing OpenRouter judge runs unchanged and Jev runs
  concurrently. Verdicts are stored under `stats.judge`;
  `correctness_score` is untouched. Any Jev failure (API error, timeout,
  exception) is logged and the run and its summary proceed as before.
- **primary**: Jev is awaited first. Its verdicts are appended to the summary
  prompt as a short `Verdicts:` block so the prose is consistent with them,
  and `correctness_score` is set to `derived_correctness_score`. If Jev fails,
  the prompt and score fall back to the shadow behaviour for that run.

`correctness_score` consumers (library `avg_correctness_score`, admin
execution analytics, `data/analytics.py`, the copilot agent generator, and
`RunSummary.tsx`) read the same field in every mode.

## Local proof without the full stack

`autogpt_platform/backend/scripts/judge_one_execution.py` builds a fixture
execution summary through the real `_build_execution_summary` and calls Jev
once with the configured key, printing the verbatim request/response, latency
and tokens:

```bash
cd autogpt_platform/backend
poetry run python scripts/judge_one_execution.py            # fixture: successful run
poetry run python scripts/judge_one_execution.py --failed   # fixture: failed run
```

`--execution <graph_exec_id> --user <user_id>` judges a real execution when a database is
reachable.
