# Activation Metrics & Experiments

How the platform measures activation, retention and unit economics across
every surface (classic agents, Autopilot, experts, schedules), and how to run
an A/B/C experiment whose results can be sliced in both PostHog and Looker.

This page is the contract between engineering and go-to-market: the event
names, the SQL views and the definitions below are the ones dashboards should
be built on. Change a definition here and in the view, and everything
downstream follows.

## Where each tool fits

| Tool | Owns | Reads |
| --- | --- | --- |
| **Postgres `analytics.*` views** | The numbers. Every metric below is a view in `autogpt_platform/analytics/queries/`. | Primary tables (`platform.*`, `auth.*`). |
| **Looker Studio** | GTM dashboards. | Only `analytics.*` views, via the `analytics_readonly` role. |
| **PostHog** | Funnels, session replay, experiment bucketing and significance testing. | Frontend autocapture + the server-side activation events below. Can also read `analytics.*` as a data-warehouse source. |
| **LaunchDarkly** | Feature gating (on/off, rollouts, targeting). Not for A/B measurement. | — |
| **DataFast** | Marketing attribution: which channel brought a visitor who signed up, ran an agent, paid. | Browser goals only (`signup`, `paywall_view`, `run_agent`, `schedule_agent`, ...). |
| **Admin dashboard** | Ops views: platform costs, copilot usage export, execution accuracy. | Backend admin routes. |
| **Sentry** | Errors and stack traces behind `agent_run_failed`. | — |

## Definitions

- **Task**: a unit of work a person asked for *now*. Either a human-started
  agent run (`triggerSource` in `manual`, `api`) or a human chat turn (a
  `user` message in an Autopilot or expert session whose origin is not
  `automation`). A run the copilot tool started (`triggerSource` = `copilot`)
  is counted once, through the chat turn that asked for it. Dry runs, nested
  sub-graph runs and dream/memory sessions are never tasks.
- **Automated task**: a schedule fire, a webhook trigger, or a scheduled
  follow-up turn. Counted separately so "people doing things" and "agents
  running" can be compared.
- **Activated**: at least 3 tasks on at least 2 distinct days within 14 days
  of signup. (`analytics.user_lifecycle.activated`)
- **Last active**: latest of last task, last visit, last scheduled run,
  falling back to signup.
- **Stale**: no activity in 14 / 30 days (`stale_14d`, `stale_30d`).
- **Churned (30d)**: had at least one task ever, signed up more than 30 days
  ago, no activity in the last 30 days.
- **Never activated (30d)**: signed up more than 30 days ago and never did a task.
- **Cost to us**: `PlatformCostLog.costMicrodollars`, our real provider spend,
  split into agent (block runs), copilot (turns) and background (dream passes).
- **Credits charged**: `CreditTransaction` `USAGE` rows, in cents.

## Activation events (PostHog)

Emitted server-side by `backend/util/product_analytics.py` with
`distinct_id = user id`, so they join the frontend's `posthog.identify`.
Every event carries `environment` and `source: "platform"`.

| GTM asked for | Event | Fires when | Key properties |
| --- | --- | --- | --- |
| run_agent | `run_agent` | A person starts an agent run (UI, API key, copilot tool). Not schedules, webhooks, sub-graphs or dry runs. | `trigger`, `trigger_ref`, `graph_id`, `graph_exec_id` |
| run_autopilot | `run_autopilot` | A person sends a message in an Autopilot chat. | `session_id`, `surface` (web chat, slack, telegram, discord) |
| run_expert | `run_expert` | A person sends a message in an expert chat (`kind: chat_turn`) or starts an expert workflow run (`kind: workflow_run`). | `expert_id`, `kind` |
| schedule_agent_run / schedule_autopilot_run / schedule_expert_run | `schedule_created` | Any schedule is registered, from any surface. | `target`: `agent` / `autopilot` / `expert`, `cron`, `is_recurring`, `schedule_id` |
| schedule_agent_ran / schedule_autopilot_ran / schedule_expert_ran | `schedule_fired` | A schedule produced work. | `target`, `schedule_id`, `graph_exec_id` or `session_id` |
| (trigger) | `trigger_fired` | A webhook produced a run. | `webhook_id`, `graph_exec_id`, `target` |
| agent_fail | `agent_run_failed` | A run reaches FAILED. | `trigger`, `failure_reason`, `expert_id` |
| — | `agent_run_completed` | A run reaches COMPLETED. | `trigger`, `cost_cents`, `duration_seconds` |
| — | `expert_hired` | A user hires an expert from a template. | `expert_id`, `template_id` |
| — | `integration_connected` | A user connects a credential, by OAuth or by pasting a key. | `provider`, `credential_type`, `method` |
| agent_idle, stale account | *(not events)* | Computed states, see `agent_health` and `user_lifecycle`. | — |

The pre-existing copilot events (`copilot_message_sent`, `copilot_tool_called`,
...) and billing events (`credit_topup_success`, `subscription_*`) are unchanged.

## SQL views (Looker)

New or extended in this change. Existing views (`retention_login_*`,
`retention_execution_*`, `users_activities`, `platform_cost_log`, ...) are
untouched.

| View | Grain | Answers |
| --- | --- | --- |
| `graph_execution` (extended) | one row per agent run | Now also `triggerSource`, `triggerRef`, `expertId`, `failureReason`, `isDryRun`, `isSubgraphRun`. |
| `chat_turn` | one row per human chat message | Autopilot vs expert usage; the SQL twin of `run_autopilot` / `run_expert`. |
| `user_task_daily` | user × day | How many tasks people run, by surface and by how they started; failures; our cost; credits charged; logins. |
| `user_lifecycle` | one row per user | First/last activity per surface, first-two-weeks behaviour, cost and revenue, and the labels `activated`, `stale_*`, `churned_30d`, `never_activated_30d`. The feature table for churn analysis. |
| `user_lifecycle_funnel_weekly` | signup-week cohort | Signup → onboarded → first task in 7d → activated in 14d → schedule / expert / purchase → retained at week 4. |
| `retention_task_weekly` | cohort × lifetime week | Retention where "active" means any human task on any surface. |
| `unit_economics_monthly` | user × month | Cost to us per user per month, per task and per active day; credits charged; gross margin; by tier. |
| `agent_health` | user × library agent | Idle, never-run and failing agents. |
| `experiment_assignment` | user × experiment | The arm each user saw; join on `user_id` to split anything else by variant. |
| `user_attribution` | one row per user | Where the user came from: shared anonymous id, PostHog device id, DataFast visitor id, first landing page, UTM tags, signup method. Join to `user_lifecycle` for activation by channel. |

Apply them to a database with:

```bash
cd autogpt_platform/backend
poetry run analytics-views
```

Then add each new view as a data source in Looker Studio (the header of every
`.sql` file documents its columns and example queries). `triggerSource` is
NULL on runs created before September 2026; the views expose those as
`agent_runs_untagged` and count them as human tasks.

## One identity across LaunchDarkly, PostHog and the database

The two tools used to disagree about who a visitor is. LaunchDarkly only
knew a user after login (every logged-out visitor shared the literal key
`anonymous`, so it could not bucket pre-signup at all), while PostHog minted
its own anonymous id and merged it into the user id at identify. Nothing
tied a pre-signup arm to a post-signup person, and nothing tied either to
the DataFast visitor that marketing sees.

Now the frontend owns one first-party anonymous id
(`src/services/analytics/anonymous-id.ts`, stored in localStorage, adopting
an existing PostHog device id when there is one) and hands it to everything:

- **PostHog** is bootstrapped with it as the anonymous distinct id, so
  `identify(user.id)` merges it into the person as usual.
- **LaunchDarkly** gets it as the anonymous `user` key before login, and as
  a `device` context alongside the `user` context after login. Rules on the
  `user` kind are unchanged; a rule that buckets by `device` keeps the same
  arm across signup.
- **The database** stores it on the user at first login (`UserAttribution`,
  reported by the browser to `POST /api/analytics/attribution`) together
  with the PostHog device id, the DataFast visitor and session ids (from the
  request headers the frontend already sends), the first landing page, UTM
  tags and the signup method. `analytics.user_attribution` exposes it.

Rule of thumb: **LaunchDarkly gates, PostHog measures, the database joins.**
Server-side flag evaluation stays keyed by user id (the backend has no
device context yet), so bucket pre-login experiments on the client.

## Running an experiment

1. Create a **multivariate feature flag** in PostHog with the arm keys you want
   (`control`, `variant-a`, `variant-b`, ...) and make it an experiment there.
2. In the frontend, read it through `useExperiment(key)` from
   `src/services/experiments/useExperiment.ts`. It returns `{ variant, isResolved }`:
   render the control experience until `isResolved`, and do not fire one-shot
   side effects before then, or a late variant is mis-recorded as control.
   The hook reports the arm to `POST /api/experiments/assignments` once per
   user and experiment; the backend keeps the **first** arm it sees.
3. Prefer a LaunchDarkly flag for the arms? Use `useLaunchDarklyExperiment(flagKey)`
   with a string-valued flag. It reports the assignment the same way
   (`source = launchdarkly`) and sends PostHog an `experiment_exposed` event
   carrying `$feature/<flag>`, so a PostHog experiment can use it as its
   exposure and measure the activation events against it.
4. Backend-decided experiments can call
   `backend.data.experiments.record_assignment(user_id, key, variant, source="backend")`.
5. Read results in PostHog (exposure + the activation events above as goals),
   and in Looker by joining `analytics.experiment_assignment` to
   `user_lifecycle`, `user_task_daily` or `unit_economics_monthly`.

The onboarding paywall experiment (`subscription-pricing-page-initial-state`)
is already wired this way.

## Deploying this change

1. Run the migrations `20260902120000_add_execution_trigger_source` (adds
   `AgentGraphExecution.triggerSource` / `triggerRef`),
   `20260902120100_add_experiment_assignment` (the `ExperimentAssignment`
   table) and `20260902120200_add_user_attribution` (the `UserAttribution`
   table). Existing `analytics.*` views are unaffected: no column they select
   from changes type.
2. Deploy the backend. From that moment every run row carries its trigger and
   every activation event flows to PostHog.
3. `poetry run analytics-views` against production, then wire the new views
   into Looker Studio.
4. Confirm in PostHog that `run_agent`, `run_autopilot`, `schedule_created`
   and `agent_run_failed` are arriving with `source = platform`.

## The questions GTM will ask next

The list GTM gave us is the first layer. These are the questions that follow
once they see the data, with where the answer already lives or what is still
missing.

**Activation**

- *Which first action predicts retention: an agent run, an Autopilot chat, an
  expert, or a schedule?* — `user_lifecycle` has the first-time timestamps
  and early counts per surface; group by the earliest one and compare
  `churned_30d`.
- *How long from signup to first task, and where do people stall?* —
  `hours_to_first_task` plus the onboarding funnel view; combine with PostHog
  session replay on users with `never_activated_30d`.
- *Does connecting an integration change activation?* —
  `integrations_connected_total` and `first_integration_connected_at` in
  `user_lifecycle` (from `IntegrationCredential`), `connected_integration` in
  the funnel, and the `integration_connected` event in PostHog.
- *Which marketing channel brings activated users?* — join
  `user_attribution` (UTM tags, referrer, landing page, DataFast visitor id)
  to `user_lifecycle`. For channels only DataFast knows, export DataFast by
  visitor id and join on `datafast_visitor_id`.

**Retention and churn**

- *Are we retaining people, or just their schedules?* — `retention_task_weekly`
  (people) next to `retention_execution_weekly` (all runs incl. automated).
- *Which early behaviours separate churned from retained users?* — the
  example query on `user_lifecycle`; the next step is a proper model, which
  needs the same table exported to a notebook.
- *Who is about to churn?* — `stale_14d` users with `tasks_28d` dropping vs
  their prior 28 days; `agent_health.idle_7d` and `failing` are the earliest
  per-agent signals. **Gap**: a weekly "at-risk" list is a view away, but
  acting on it (email, in-product nudge) needs the alerts pipeline.
- *Do failures cause churn?* — `agent_runs_failed_total` and
  `agent_runs_no_credits_total` in `user_lifecycle`; `failure_reason` in
  `graph_execution`. Insufficient balance is already separated because it is
  a pricing problem, not a product problem.

**Cost and pricing**

- *What will a free trial with card on file cost us?* —
  `unit_economics_monthly` by tier and by month-of-life; the
  `PERCENTILE_CONT` example shows the tail, which is what a trial cap must
  cover.
- *Cost per task by surface and model?* — split exists by surface in
  `user_task_daily`; per-model cost is in `platform_cost_log`. **Gap**: chat
  turns are not linked to the exact cost rows (copilot cost rows carry the
  session id in `graphExecId`), so cost per *turn* is an average, not exact.
- *Gross margin per user and per tier?* — `gross_margin_usd` in
  `unit_economics_monthly`, with the caveat that credits are prepaid.
- *How much do we spend on users who never come back?* — join
  `unit_economics_monthly` to `user_lifecycle.churned_30d`.
- *Background cost (dream/memory passes) as a share of total?* —
  `background_cost_usd` in `user_task_daily` and `unit_economics_monthly`.

**Experiments**

- *Did the pricing-page arm change activation or retention, not just
  conversion?* — `experiment_assignment` joined to `user_lifecycle`.
- *Can we A/B backend behaviour (model routing, default expert, prompt)?* —
  yes via `record_assignment(source="backend")`; LaunchDarkly stays the
  gate, PostHog and the assignment table hold the arm.
- *Can the team keep using LaunchDarkly for arms?* — yes, through
  `useLaunchDarklyExperiment`; the arm still lands in PostHog and the
  assignment table. **Gap**: the backend's LaunchDarkly context has no
  `device` kind yet, so server-evaluated flags cannot bucket by the
  anonymous id.

**Experts and schedules**

- *Do hired experts get used after day one?* — `expert_turns_total`,
  `expert_workflow_runs_total`, `experts_active` in `user_lifecycle`;
  `expert_hired` event in PostHog for the funnel.
- *How many schedules are created and then silently stop?* — creations are in
  `ActivityEvent`, fires are in `graph_execution.triggerSource = 'schedule'`.
  **Gap**: schedules live only in the APScheduler job store, so there is no
  durable schedule row to join pauses and deletions to; a `Schedule` table
  would make schedule health first-class.

**Operational gaps to close next**

- No `lastActiveAt` on `User`; `user_lifecycle` derives it, which is fine for
  dashboards but not for real-time gating.
- Frontend PostHog is not bootstrapped with flags, so experiment variants
  resolve after first paint. `useExperiment.isResolved` papers over this;
  server-side bootstrap would remove the delay.
- DataFast goals fire only after analytics consent, so they undercount
  relative to the server-side events. Use the server-side numbers as truth
  and DataFast for channel attribution only.
