# Capy Create Automation
<!-- MANUAL: file_description -->
Set up standing work for Capy's coding agents: an automation starts a run whenever its trigger fires.
<!-- END MANUAL -->

## Capy Create Automation

### What it is
Sets up a Capy automation: a standing job that starts a Capy coding agent run on a schedule or when an event arrives from GitHub, Sentry, Linear, Slack or a webhook, e.g. open a fix pull request for every new Sentry error. Runs are capped per day and show up as ordinary Capy threads.

### How it works
<!-- MANUAL: how_it_works -->
Calls `POST /api/v1/automations` with one trigger built from the inputs. `schedule` takes `cron` and `timezone`. `github`, `slack`, `sentry` and `linear` take an `event`, optional `conditions`, and a `run_when` sentence Capy checks each event against before it starts a run. `incoming_webhook` returns a `webhook_url` to POST events to, and `on_demand` runs only when started by hand. A missing cron or an unknown event fails before Capy is called, with the valid choices in the error.

Each run starts from `prompt` in the project's repositories and shows up as a Capy thread, so Capy List Threads and Capy Wait For Thread can follow it. `max_runs_per_day` (10 by default) caps how many runs Capy starts in a day, and `request_id` makes the create idempotent, so a retried call returns the same automation.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| project_id | The Capy project whose repositories the runs work in (see Capy List Projects) | str | Yes |
| name | A short name for the automation | str | Yes |
| prompt | The brief every run starts from, written for its trigger, e.g. 'Root-cause this Sentry issue and open a fix pull request if it is fixable' | str | Yes |
| trigger | What starts a run: schedule (a cron), github, slack, sentry or linear events, incoming_webhook (a URL you POST to), or on_demand (started by hand) | "schedule" \| "github" \| "slack" \| "sentry" \| "linear" \| "incoming_webhook" \| "on_demand" | Yes |
| event | For github, slack, sentry and linear: the event that starts a run. github: pull_request_opened, pull_request_merged, checks, workflow_run, issue_comment, label_change and more. slack: message or reaction. sentry: any_issue, issue_lifecycle or event_alert. linear: issue_created or status_changed. | str | No |
| cron | For schedule: a five-field cron, e.g. 0 9 * * 1-5 | str | No |
| timezone | For schedule: the IANA timezone the cron runs in | str | No |
| run_when | One plain sentence Capy checks each event against before it starts a run, e.g. 'Only errors raised by the backend'. Not used by schedule or on_demand. | str | No |
| conditions | Filters on the event, in Capy's shape for the trigger; an event passes a filter when it matches any listed value. github: repositories, branches, labels, authors, conclusions. sentry: projects, levels. linear: teams, labels, statuses. slack: channels and users, by id. incoming_webhook: contains, excludes, regex. | Dict[str, Any] | No |
| max_runs_per_day | The most runs Capy starts in a day, so a noisy trigger can't run up the bill | int | No |
| model_id | Capy model ID every run uses, e.g. openai/gpt-6-astra, or a bare name to combine with model_route. Empty lets each run use its owner's default model. | str | No |
| model_route | Who pays for the model. as_given uses model_id exactly as written. capy_balance bills the Capy balance. codex, copilot, supergrok and azure run the model through that provider linked in Capy's settings, so it bills the subscription instead. | "as_given" \| "capy_balance" \| "codex" \| "copilot" \| "supergrok" \| "azure" | No |
| reasoning | Reasoning effort for model_id. Needs model_id. | "" \| "none" \| "instant" \| "minimal" \| "low" \| "medium" \| "high" \| "xhigh" \| "max" | No |
| thread_mode | new starts a thread per run; single keeps every run in one thread, so each run sees the earlier ones | "new" \| "single" | No |
| machine_size | Machine size for each run's VM. Empty uses Capy's default. | "" \| "small" \| "medium" \| "large" \| "ultra" \| "hyper" \| "bigguy" | No |
| description | What the automation is for | str | No |
| enabled | Start listening right away. False creates it paused. | bool | No |
| request_id | Idempotency key. Re-sending the same request_id returns the automation it already created. Leave empty to generate one. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| automation | The created automation | Automation |
| automation_id | Pass to Capy Set Automation Enabled or Capy Delete Automation | str |
| enabled | Whether it is listening for its trigger | bool |
| webhook_url | For an incoming_webhook trigger: POST events here to start runs. Keep it private, since every request can start a paid run. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Sentry Triage**: Start a run for each new Sentry issue that root-causes it and opens a fix pull request when it can.

**CI Fixer**: Start a run whenever a pull request's checks fail, to diagnose the failure and push a fix.

**Nightly Maintenance**: Run dependency updates or a flaky-test sweep on a schedule and open a pull request with the result.
<!-- END MANUAL -->

---
