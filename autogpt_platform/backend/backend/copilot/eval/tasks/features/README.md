# Task features eval

States what we expect from two task features and shows whether it holds:
task controls (start, stop, try again) and suggested next tasks in the
morning brief. It reuses the task eval's simulated companies
(`../data/worlds`), engines, simulator and reviewers, and writes the same
kind of report.

```
data/cases/controls.json      start, stop, try again; the worker's turn
data/cases/suggestions.json   accounts that get next tasks drafted
```

22 cases: controls has 11 (4 code-only, 4 model, 3 live) and suggestions
has 11 (5 code-only, 5 model, 1 live).

## Running it

```
# list the cases and what a real run would cost; no call of any kind
poetry run python -m backend.copilot.eval.tasks.features --dry-run \
    --reviewer claude:claude-opus-5-5 --reviewer cursor:grok-4.7-medium-fast

# code-only cases: no model, no database, free
poetry run python -m backend.copilot.eval.tasks.features --offline

# everything except the live tier
poetry run python -m backend.copilot.eval.tasks.features \
    --backend claude-native --model claude-opus-5-5 --user-model claude-sonnet-5-5 \
    --drafter claude:claude-haiku-5-5 \
    --reviewer claude:claude-opus-5-5 --reviewer cursor:grok-4.7-medium-fast

# add the live tier (Postgres, Redis, RabbitMQ and a copilot executor running)
... --live --live-user-id <an existing dev user id>
```

`--area controls|suggestions`, `--case` and `--needs none|model|live` pick
cases. A run writes `runs/<timestamp>/` with `results.jsonl`,
`summary.json`, `report.md` and `report.html`.

## What a case needs

- `none`: code only. A controls case checks the real `run_prompt` and
  settle logic on a task row (`records.py` builds it the way production
  stores it); a suggestions case runs the real `draft_task_suggestions`
  with a scripted model reply.
- `model`: a worker turn on the simulated world (the task eval's engines,
  with production's `run_prompt` as the first message), or a real drafting
  call (`--drafter`, production uses the aux tier, Haiku 5.5).
- `live`: real tasks for `--live-user-id`, started, retried and stopped
  through `copilot.task_run`, waited on until the executor settles them.
  Each start is a paid worker turn on the platform's models. Every task a
  live case makes is archived when it ends.

`review: true` asks every `--reviewer` for a judgement; the case then
scores half on code checks and half on the reviewer, as in the task eval.
A case without one scores on its checks alone and passes only if all pass.

## Case

```jsonc
{
  "id": "controls_003",                     // <area>_<nnn>
  "area": "controls",                       // controls | suggestions
  "world": "saffron_kitchens",              // a task eval world
  "title": "…one line for the report…",
  "needs": "none",                          // none | model | live
  "review": true,
  "task": {…}, "note": "…",                 // controls
  "account": { "board": [], "live": [], "dismissed": [], "retired": [] },   // suggestions
  "scripted_reply": [ {…draft…} ],          // stands in for the drafting model
  "live": "start",                          // start | retry | stop | start_suggestion
  "expected": {
    "must": ["…plain words…"], "must_not": ["…"],
    "checks": { … }                         // models.FeatureChecks; left out = not checked
  }
}
```
