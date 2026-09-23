# PostHog Tracking Plan

The single list of PostHog events: for every user action, one event name, one
sender and one set of properties. [Activation Metrics & Experiments](activation-metrics.md)
explains how PostHog fits next to the SQL views, Looker, LaunchDarkly and
DataFast; this page is the event contract.

Event names follow the product analytics plan, *Every Second Counts* (owner:
Toran), section "What we will record". Where that plan names an event for an
action, this list uses its name; [Differences from the analytics plan](#differences-from-the-analytics-plan)
lists what is not aligned yet.

In code the names live in two modules, and nothing outside them spells out an
event name:

| Sender | Module | Pin test |
| --- | --- | --- |
| Backend | `autogpt_platform/backend/backend/util/posthog_events.py` (`PostHogEvent`, `PlannedPostHogEvent`) | `posthog_events_test.py` |
| Browser | `autogpt_platform/frontend/src/services/analytics/posthog-events.ts` (`PostHogEvent` and its groups, `PlannedPostHogEvent`) | `__tests__/posthog-events.test.ts` |

## Rules

### Naming

- **Use the analytics plan's name.** When the plan names an event for an
  action, that is the name, even if we used to send something else.
- Otherwise the name is `object_action`, snake_case, action in the past
  tense: `signup_completed`, `checkout_started`, `paywall_viewed`.
- Chat (Autopilot) events start with `chat_`, never `copilot_`. The trial
  lifecycle is `trial_*`.
- **Rename once, then never again.** PostHog stores the raw string, so a
  rename empties every insight, funnel and cohort built on the old name, and
  past events cannot be backfilled. The live names that differ from the plan
  (status `rename → X`) are renamed together in SECRT-2722, which lists
  every old → new pair for the insights to update. After that, the pin tests
  fail on any change to a live name.
- Put a new name in `PlannedPostHogEvent` when it is agreed, and move it into
  the live list in the change that starts sending it.

### Who sends it

- **Outcomes are sent by the backend**: something was created, charged, run,
  hired or delivered. The backend sees every path (web, API, copilot, Slack,
  Telegram, webhooks), cannot be blocked by an ad blocker, and is what the
  `analytics.*` SQL views count.
- **Interactions are sent by the browser**: views, clicks, dismissals and
  client-side timings that the backend never sees.
- One action has **one** event. Where two emitters describe the same action,
  the backend event survives and the other is merged into it (status
  `merge`).

### Base properties

Every event must carry:

| Property | Value |
| --- | --- |
| `environment` | `settings.config.app_env` on the backend (`local`, `dev`, `prod`), the matching app environment in the browser. |
| `source` | Which emitter sent it. Existing values stay: `platform` (`product_analytics.py`), `chat_copilot` (`copilot/tracking.py`). Newly covered backend emitters use `platform`; the browser uses `web`, registered once as a PostHog super property. |

Today only `product_analytics.py` and `copilot/tracking.py` send them. The
funnel (`funnel_analytics.py`), billing (`data/credit.py`) and trial
(`notifications/trial.py`) events and every browser event carry neither;
SECRT-2722 adds them.

Three events already use `source` for something else and must move that
value to its own property before the base property is applied, or the two
will overwrite each other: `hire_started` (`source: onboarding_hire_step`),
`expert_recommended` (the team's `source`) and `workflow_installed_on_expert`
(`source: library | marketplace`).

### Identity

- **`distinct_id` is the platform user id** for every backend event, the same
  id the browser passes to `posthog.identify(user.id)`, so both halves land
  on one person.
- **Before login the browser uses the first-party anonymous id**
  (`src/services/analytics/anonymous-id.ts`), passed to `posthog.init` as the
  bootstrap `distinctID`. `identify` merges it into the user at login, and
  `resetAnalyticsIdentity` mints a fresh one at logout.
- **No synthetic ids.** A backend event with no user is dropped, as
  `product_analytics.track` already does. `copilot/tracking.py` still falls
  back to `anonymous_{session_id}` for `copilot_message_sent` and
  `copilot_tool_called`; each of those creates a person nobody can merge, and
  SECRT-2722 removes the fallback.
- **No email or name in event properties.** Identify ids only. The email and
  display name are person properties set by `identify` and nowhere else.
  (`expert_hired.name` is the expert's display name, not a person's.)

## What counts as a task

The `analytics.*` views (`user_task_daily`, `user_lifecycle`,
`retention_task_weekly`, `unit_economics_monthly`) define a task as a unit of
work a person asked for now:

- a top-level, non-dry-run agent run whose `triggerSource` is `manual` or
  `api` (or NULL, for rows older than the column), expert workflow runs
  included; **or**
- a `user` message in an Autopilot or expert chat whose session origin is not
  `automation` and whose kind is not `dream`.

A run the copilot started (`triggerSource = copilot`) is **not** a second
task: the chat turn that asked for it already counts. `run_agent` does fire
for copilot-started runs (with `trigger: copilot`), because it measures
human-initiated runs, not tasks. `user_lifecycle.agent_runs_human_total` is
its SQL twin and includes copilot runs too.

The same definition as a PostHog filter (with today's names; after
SECRT-2722 it reads `agent_run_started` and `chat_message_sent`):

```sql
event IN ('run_agent', 'run_expert', 'run_autopilot')
AND coalesce(properties.trigger, '') != 'copilot'
```

In the insight UI: events `run_agent`, `run_expert` and `run_autopilot`,
filtered by `trigger` "is not" `copilot`. Chat turns carry no `trigger`, so
the filter must keep events where it is unset. The emitters already leave
out dry runs, sub-graph runs, schedule and webhook runs, and automation-origin
turns.

Automated work is measured separately: `schedule_fired` and `trigger_fired`.

## Status legend

| Status | Meaning |
| --- | --- |
| `live` | Sent today under this name. Keep it. |
| `rename → X` | Sent today under an old name. SECRT-2722 renames it to `X`, the analytics plan's name. |
| `merge → X` | Sent today, duplicates `X`. SECRT-2722 stops sending it once dashboards read `X`; the name stays reserved. |
| `add` | Not sent yet. SECRT-2723 adds it; the name is already in `PlannedPostHogEvent`. |
| `remove` | Declared in code but never sent. SECRT-2722 deletes it. |

## Acquisition

| Event | Sender | Status | Required properties | Fires when |
| --- | --- | --- | --- | --- |
| `$pageview` | browser | live | `$current_url` | A route or query string changes (`PostHogPageViewTracker`). |
| `tour_started` | browser | add | — | The public `/tour` page is opened (once per tab). |
| `tour_scenario_started` | browser | add | `scenario` | A tour scenario starts playing. |
| `tour_scenario_completed` | browser | add | `scenario` | A tour scenario reaches its end. |
| `tour_cta_clicked` | browser | add | `label` (`pricing`, `another-scenario`, `self-host`, `share`) | A tour call to action is clicked. |
| `signup_completed` | backend | add | `signup_method` | The user row is created. |

The tour funnel is sent to DataFast today (`tour_start`, `tour_scenario_start`,
`tour_scenario_complete`, `tour_cta_click`); the PostHog events mirror it.

## Onboarding and activation

| Event | Sender | Status | Required properties | Fires when |
| --- | --- | --- | --- | --- |
| `onboarding_step_viewed` | browser | add | `step` (`team`, `autopilot`, `role`, `pain_points`, `connect`, `hire`, `preparing`) | A wizard step is shown (once per tab, same keys as the DataFast `onboarding_<step>` goals). |
| `onboarding_completed` | backend | add | — | The onboarding is marked complete. |
| `brain_dump_started` | browser | live | — | Recording starts. |
| `brain_dump_completed` | browser | live | `input_mode`, `duration_secs` (voice) or `chars` (typed) | A dump was accepted and the wizard advances. |
| `brain_dump_canceled` | browser | live | — | Recording is cancelled. |
| `brain_dump_skipped` | browser | live | — | The step is skipped. |
| `brain_dump_recovery_shown` | browser | live | `parts` | A saved partial recording is offered back. |
| `brain_dump_recovery_used` | browser | live | — | The saved recording is used. |
| `brain_dump_retry` | browser | live | `attempt` | A failed finalize is retried. |
| `brain_dump_restarted` | browser | live | — | Recording starts over. |
| `brain_dump_download` | browser | live | — | The recording is downloaded after a failure. |
| `brain_dump_permission_denied` | browser | live | — | Microphone permission is refused. |
| `brain_dump_typed_fallback` | browser | live | `reason` | The user types instead of speaking. |
| `transcription_failed` | browser | live | `error_code` | Finalize returns a failure. |
| `finalize_latency_ms` | browser | merge → `brain_dump_completed` | `ms`, `input_mode` | Finalize returns. Becomes a `finalize_latency_ms` property on `brain_dump_completed` and `transcription_failed`. |
| `welcome_dialog_closed` | browser | live | — | The first-landing welcome dialog closes. |
| `capability_card_viewed` | browser | live | `card_index`, `deck` | A capability card is shown. |
| `capability_cards_completed` | browser | live | `card_index`, `deck` | The deck is finished. |
| `capability_cards_skipped` | browser | live | `card_index`, `deck` | The deck is skipped. |
| `intro_path` | browser | live | `path` | The intro path (A/B) is chosen. |
| `intro_followup_sent` | browser | live | `chars` | First message after the intro card. |
| `later_dump_completed` | browser | live | — | A brain dump sent later from the composer. |
| `expert_recommended` | browser | live | `template_id`, `position`, `source` (see base properties) | A recommended expert card is shown. |
| `expert_recommendation_clicked` | browser | live | `template_id`, `position` | A recommended expert card is clicked. |
| `hire_step_continued` | browser | live | `hired`, `recommended` | The hire step is left. |
| `onboarding_expert_hired` | browser | merge → `expert_hired` | `template_id`, `position` | A hire lands from the onboarding hire step. |
| `intro_card_dismissed` | browser | remove | — | Never sent. |
| `raise_door_clicked` | browser | remove | — | Never sent. |
| `intro_start_with_autopilot` | browser | remove | — | Never sent. |
| `tab_intro_shown` | browser | live | `tab` | A tab intro card is shown. |
| `tab_intro_cta_clicked` | browser | live | `tab`, `cta` | Its primary CTA is used. |
| `tab_intro_dismissed` | browser | live | `tab` | It is dismissed any other way. |
| `integration_connected` | backend | live | `provider`, `credential_type`, `method` | A credential is stored (OAuth, key, device code). |
| `credential_card_never_rendered` | browser | live | `provider`, `failure_class` | A provider is missing from the provider map. |
| `credential_oauth_popup_blocked` | browser | live | `provider`, `failure_class` | Both the popup and the new tab were blocked. |
| `credential_oauth_flow_timed_out` | browser | live | `provider`, `failure_class` | The OAuth flow timed out. |
| `credential_scope_shortfall_blocked_selection` | browser | live | `provider`, `failure_class` | The granted scopes are narrower than required. |
| `credential_proceed_stuck_after_connect` | browser | live | `provider`, `failure_class` | A connected credential never reached the card. |

## Engagement

| Event | Sender | Status | Required properties | Fires when |
| --- | --- | --- | --- | --- |
| `run_agent` | backend | rename → `agent_run_started` | `graph_id`, `graph_exec_id`, `trigger` (`manual`, `api`, `copilot`), `trigger_ref`, `preset_id` | A person starts a non-expert agent run. |
| `run_autopilot` | backend | rename → `chat_message_sent` | `session_id`, `origin`, `surface`, `kind: chat_turn` | A person sends a message in an Autopilot chat. |
| `run_expert` | backend | rename → `chat_message_sent` (chat turn) / `agent_run_started` (workflow run) | `expert_id`, `kind` (`chat_turn` or `workflow_run`); chat: `session_id`, `origin`, `surface`; run: `graph_id`, `graph_exec_id`, `trigger` | A person messages an expert or starts an expert workflow. |
| `copilot_message_sent` | backend | merge → `chat_message_sent` | `session_id`, `message_length` | Same chat turn. `message_length` moves onto `chat_message_sent`. |
| `agent_run_completed` | backend | rename → `agent_run_finished` (`status: completed`) | `graph_id`, `graph_exec_id`, `trigger`, `expert_id`, `cost_cents`, `duration_seconds` | A run reaches COMPLETED (sub-graph and automated runs included). |
| `agent_run_failed` | backend | rename → `agent_run_finished` (`status: failed`) | as above plus `failure_reason` | A run reaches FAILED. |
| `expert_run_completed` | backend | merge → `agent_run_finished` | `expert_id`, `status`, `graph_exec_id` | A top-level expert run reaches COMPLETED or FAILED. Same as `agent_run_*` with `expert_id` set on a top-level run. |
| `copilot_agent_run_success` | backend | rename → `chat_outcome` (`outcome_type: agent_run_success`) | `session_id`, `graph_id`, `graph_name`, `execution_id`, `library_agent_id` | The copilot started a run for the user: a moment of value in the chat. |
| `schedule_created` | backend | live | `schedule_id`, `target` (`agent`, `autopilot`, `expert`), `expert_id`, `cron`, `is_recurring`, `run_at`, `graph_id`, `session_id` | Any schedule is registered, from any surface. |
| `copilot_agent_scheduled` | backend | rename → `chat_outcome` (`outcome_type: schedule_created`) | `session_id`, `graph_id`, `schedule_id`, `cron`, ... | The copilot scheduled an agent. `schedule_created` (`target: agent`) still counts the schedule itself. |
| `copilot_followup_scheduled` | backend | rename → `chat_outcome` (`outcome_type: schedule_created`) | `session_id`, `schedule_id`, `is_recurring` | The copilot scheduled its own follow-up. `schedule_created` (`target: autopilot` or `expert`) still counts the schedule itself. |
| `copilot_tool_called` | backend | rename → `chat_tool_called` | `session_id`, `tool_name`, `tool_call_id` | The copilot calls a tool. |
| `copilot_library_check_outcome` | backend | rename → `chat_library_check_outcome` | `session_id`, `outcome`, `matches_count`, `top_score` | The create-agent library check ends. |
| `copilot_trigger_setup` | backend | remove | — | `track_trigger_setup` has no caller. |
| `voice_mode_started` | browser | live | `entry` | Voice mode is switched on. |
| `voice_mode_stopped` | browser | live | `turns`, `state` | Switched off by the user. |
| `voice_mode_timed_out` | browser | live | `turns`, `state` | Closed by the silence timeout. |
| `voice_turn_sent` | browser | live | `turn_index`, `transcript_chars` | A spoken turn is sent. |
| `voice_turn_dropped` | browser | live | `reason` | A spoken turn is discarded. |
| `voice_turn_completed` | browser | live | `turn_index` | The mic reopens after the reply. |
| `voice_transcribe_retried` | browser | live | `turn_index` | A failed transcription is retried. |
| `voice_recording_downloaded` | browser | live | `turn_index` | The audio is downloaded instead. |
| `voice_mode_permission_denied` | browser | live | `stage` | Microphone permission is refused. |
| `voice_mode_error` | browser | live | `stage` | The VAD, synthesis or send failed. |
| `voice_transcribe_latency_ms` | browser | merge → `voice_turn_sent` | `ms` | Becomes a `transcribe_latency_ms` property on `voice_turn_sent` (and on `voice_turn_dropped` with `reason: filler_or_empty`). |
| `voice_first_sound_latency_ms` | browser | merge → `voice_turn_completed` | `ms` | Becomes a `first_sound_latency_ms` property on `voice_turn_completed`. |
| `experts_section_viewed` | browser | live | — | The marketplace experts shelf renders with results. |
| `expert_profile_opened` | browser | live | `template_id` | An expert profile page opens. |
| `hire_started` | browser | live | `template_id` | A hire button is clicked. Sent twice per click from the onboarding hire step (`useHireStep.ts`); SECRT-2722 keeps one. |
| `hire_flow_abandoned` | browser | live | `template_id`, `stage` | The hire dialog is closed before hiring. |
| `expert_hired` | backend | live | `expert_id`, `template_id`, `name` | An expert is hired. SECRT-2722 moves it into `hire_expert` so an idempotent re-hire of an active expert no longer fires, and adds `failed_preloads_count` and the hiring `surface`. |
| `hire_completed` | backend | merge → `expert_hired` | `template_id`, `failed_preloads_count` | Same hire, already skipping re-hires. |
| `hire_flow_completed` | browser | merge → `expert_hired` | `template_id`, `expert_id`, `elapsed_ms`, `voice_picked` | Same hire, from the marketplace hire dialog. |
| `hire_failed` | backend | live | `template_id`, `failed_preloads_count` | Hiring raised. |
| `expert_thread_created` | browser | live | `expert_id` | A new expert chat is created. |
| `writing_style_added` | backend | live | `expert_id` | A writing style is saved on an expert. |
| `workflow_installed_on_expert` | backend | live | `expert_id`, `source` (see base properties), `library_agent_id` or `store_listing_version_id` | A workflow is attached to an expert. |
| `home_viewed` | browser | live | — | The home dashboard renders with data. |
| `home_attention_actioned` | browser | live | `kind`, `action` | A "needs you" item is approved or declined. |
| `home_team_member_clicked` | browser | live | `expert_id` | A team member row is clicked. |
| `listing_added_to_library` | backend | add | `store_listing_version_id`, `graph_id`, `library_agent_id` | A marketplace agent is added to the library for the first time. |
| `listing_downloaded` | backend | add | `store_listing_version_id`, `graph_id` | A marketplace agent is downloaded. |

## Monetization

| Event | Sender | Status | Required properties | Fires when |
| --- | --- | --- | --- | --- |
| `paywall_viewed` | browser | add | `surface` (`onboarding`, `paywall_gate`, `billing`) | A paywall or plan picker is shown (once per tab per surface). The pricing arm comes from PostHog's own `$feature/...` properties. |
| `plan_selected` | browser | add | `subscription_tier`, `billing_cycle`, `surface` | A plan is picked on any paywall or on the billing page. |
| `billing_portal_opened` | browser | add | `surface` | The Stripe billing portal is opened. |
| `checkout_started` | backend | add | `checkout_kind` (`subscription`, `top_up`), `subscription_tier`, `billing_cycle`, `surface` | A Stripe Checkout session is created. |
| `checkout_abandoned` | browser | add | `checkout_kind`, `surface` | The user returns from Stripe Checkout without paying (the cancel URL). Browser-sent: Stripe only reports the expiry a day later. |
| `subscription_trial_offer_viewed` | browser | rename → `trial_offer_viewed` | `trial_offer_version`, `subscription_tier`, `trial_duration_days`, `surface` | A trial offer card is shown. |
| `subscription_trial_checkout_started` | browser | live | `trial_offer_version`, `surface` | Trial checkout is opened. Once `checkout_started` ships, fold this in as `checkout_kind: trial`. |
| `subscription_trial_started` | backend | rename → `trial_started` | `trial_id`, `trial_offer_version`, `subscription_tier`, `billing_cycle`, `trial_duration_days` | The trial starts. Like every trial lifecycle event, it is sent only when its notification email is queued. |
| `subscription_trial_ending` | backend | rename → `trial_ending` | as above | The reminder window opens. |
| `subscription_trial_canceled` | backend | rename → `trial_canceled` | as above | The trial is set to cancel. |
| `subscription_trial_resumed` | backend | rename → `trial_resumed` | as above | A cancelled trial is resumed. |
| `subscription_trial_payment_failed` | backend | rename → `payment_failed` | as above | The conversion charge fails. |
| `subscription_trial_converted` | backend | rename → `trial_converted` | as above | The trial converts to paid. |
| `subscription_trial_ended` | backend | rename → `trial_ended` | as above | The trial ends without converting. |
| `subscription_upgraded` | backend | rename → `subscription_changed` (`change_type: upgrade`) | `previous_subscription_tier`, `subscription_tier`, `billing_cycle` | A paid tier change takes effect. |
| `subscription_payment_success` | backend | rename → `payment_succeeded` | `subscription_tier`, `billing_cycle`; SECRT-2723 adds `amount_cents`, `currency` | A subscription invoice is paid. |
| `credit_topup_success` | backend | rename → `topup_completed` | `amount_credits`, `top_up_type`; SECRT-2723 adds `amount_cents`, `currency` | Credits are bought. |
| `subscription_cancellation_scheduled` | backend | live | `subscription_tier` | A paid plan is set to cancel at period end. |
| `subscription_ended` | backend | add | `subscription_tier`, `billing_cycle`, `reason` | A paid subscription ends (Stripe `customer.subscription.deleted`). |
| `subscription_tier_reconciliation_discrepancy` | backend | rename → `subscription_tier_reconciled` | `direction`, `previous_subscription_tier`, `subscription_tier`, `via` | Ops signal: Stripe and the stored tier disagreed. Not a user action; keep out of funnels. |

The onboarding paywall's `paywall_view`, `paywall_checkout_cancelled` and
`hire_completed` goals go to DataFast only, which is why the paywall has no
PostHog funnel yet.

## Retention

| Event | Sender | Status | Required properties | Fires when |
| --- | --- | --- | --- | --- |
| `schedule_fired` | backend | live | `schedule_id`, `target`, `expert_id`, `graph_id`, `graph_exec_id` or `session_id` | A schedule produces work. |
| `trigger_fired` | backend | live | `webhook_id`, `graph_id`, `graph_exec_id`, `expert_id`, `preset_id`, `target` | A webhook produces a run. |
| `briefing_generated` | backend | live | `run_count`, `decision_count`, `has_content` | A morning briefing is composed (or found empty). |
| `briefing_delivered` | backend | live | `briefing_id` | It is posted to the user's thread. |
| `briefing_opened` | browser | live | — | The briefing renders on home. Not the plan's `briefing_opened_in_chat`, which is opening it in the chat thread. |
| `briefing_outcome_clicked` | browser | live | `status` | An outcome row in the briefing is clicked. |
| `expert_fired` | backend | live | `expert_id` | An expert is fired. |

Return visits are `$pageview`; account-level retention is computed in
`retention_task_weekly` and `user_lifecycle`, not sent as events.

## Experiments

| Event | Sender | Status | Required properties | Fires when |
| --- | --- | --- | --- | --- |
| `experiment_exposed` | browser | live | `experiment_key`, `variant`, `provider: launchdarkly`, `$feature/<flag>` | A LaunchDarkly-bucketed arm is shown to a signed-in user (`useLaunchDarklyExperiment`). |
| `$feature_flag_called` | browser (posthog-js) | live | set by PostHog | A PostHog flag is read (`useExperiment`). |

Arms are also stored in the database (`analytics.experiment_assignment`),
so an experiment can be read in PostHog and Looker alike.

## Differences from the analytics plan

Left for follow-up changes, so this list and the plan can be compared line by
line:

- **Events the plan folds into others keep their names until that change.**
  The brain-dump events (`brain_dump_*`, `transcription_failed`,
  `later_dump_completed`) and the wizard's `intro_path` and
  `hire_step_continued` become properties of `onboarding_step_viewed` /
  `_completed` / `_skipped` / `_back`. Spoken turns (`voice_turn_sent`)
  become `chat_message_sent` with `input_mode: voice`.
- **Property names.** The plan's envelope says `chat_session_id` and `via`;
  chat events still send `session_id` and run events `trigger`.
- **`subscription_changed` covers upgrades only.** The plan also counts
  cancellations and downgrades there; `subscription_cancellation_scheduled`
  and `subscription_ended` stay separate events.
- **`chat_outcome` has two outcome types.** Only `agent_run_success` and
  `schedule_created` have an emitter. `agent_created`, `trigger_setup`,
  `artifact_created`, `file_produced` and `answer_only` do not.
- **Plan events we do not send yet** (`signup_started`, `chat_session_started`,
  `chat_response_completed`, `chat_blocked`, `screen_viewed`,
  `screen_engaged`, `milestone_reached`, the builder, library and email
  families, ...) come with the plan's phases, not with this list.
- **Events the plan has no name for keep their own**, e.g.
  `integration_connected`, `schedule_created`, `hire_started`,
  `billing_portal_opened`, `tour_*`, `tab_intro_*`, `voice_*` and
  `credential_*`. `briefing_opened` is the briefing shown on home, a
  different action from the plan's `briefing_opened_in_chat`.

## Events not in the constants modules

These are sent by PostHog or a third party, not by our code, and are listed
so nobody adds a duplicate:

- `$pageleave`, `$autocapture`, `$identify` and `$feature_flag_called`:
  posthog-js (`capture_pageleave` and `autocapture` are on).
- `$ai_generation`: OpenRouter's PostHog integration, for copilot title
  generation (`posthogDistinctId` in `copilot/service.py`).
