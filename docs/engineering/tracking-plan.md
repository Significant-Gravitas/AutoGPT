# PostHog Tracking Plan

The single list of PostHog events: for every user action, one event name, one
sender and one set of properties. [Activation Metrics & Experiments](activation-metrics.md)
explains how PostHog fits next to the SQL views, Looker, LaunchDarkly and
DataFast; this page is the event contract.

PostHog is the only place for what people do in the app. DataFast gets no
in-app events (SECRT-2718): it keeps its script for visits and marketing
attribution, the `/tour` goals (including `tour_cta_click` with
`surface: marketplace` from the logged-out marketplace sidebar, which shares
the tour's tracking), the server-side `signup` goal
(`datafast-server.ts`), and the DataFast visitor and session ids that ride
along to the backend (`X-Datafast-*` headers) and into Stripe Checkout
metadata, so purchases are attributed to the channel that brought the visitor.

Event names follow the product analytics plan, *Every Second Counts* (owner:
Toran), section "What we will record". Where that plan names an event for an
action, this list uses its name; [Differences from the analytics plan](#differences-from-the-analytics-plan)
lists what is not aligned yet.

In code the names live in two modules. Browser call sites may still pass a
literal (`trackBrainDump("brain_dump_started")`), which is type-checked
against these modules, and backend funnel `data_index` keys embed the name
too (`briefing_generated:<id>`, `briefing_delivered:<id>`). A
rename therefore has to search for the old string, not just edit the modules:

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
  past events cannot be backfilled. The live names that differed from the
  plan were renamed together in SECRT-2722 (see [Renamed](#renamed)); the
  pin tests reserve the old names and fail on any change to a live one.
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
  the backend event survives and the other stops being sent (see
  [Removed](#removed)).

### Base properties

Every event must carry:

| Property | Value |
| --- | --- |
| `environment` | `settings.config.app_env` on the backend (`local`, `dev`, `prod`), the matching app environment in the browser. |
| `source` | Which emitter sent it: `chat_copilot` (the events `copilot/tracking.py` captures itself; `chat_message_sent` goes through `product_analytics` and is `platform`), `platform` (every other backend emitter), `web` (the browser), `website` (the marketing site, see [Marketing site](#marketing-site)). |
| `revision` | The deployed build's git sha, which Sentry reports as its release: `SENTRY_RELEASE` on the backend, the build's commit in the browser (`SENTRY_RELEASE`, else Vercel's `VERCEL_GIT_COMMIT_SHA`). Left out of a build that has none, such as a local one, and from `$ai_generation` (below). |

Every backend event goes through `capture()` in
`backend/util/posthog_client.py`, which applies `environment` and `source`
last, so a caller cannot overwrite them. `revision` is a super property of
every backend PostHog client instead (`revision_properties()`), so the flag
client's `$feature_flag_called` carries it too. AutoPilot's `$ai_generation`
is the exception: OpenRouter builds it from the request rather than through
our client, so it carries no `revision`; the build arrives as
`metadata_revision` from `trace.revision` instead. AutoPilot's `$ai_trace` and
`$ai_span` go through `capture()`, so they carry `revision`. The
browser registers all three once as PostHog super properties
(`src/services/analytics/posthog-base-properties.ts`), again after `reset()`
at logout; a Vercel preview reports `environment: preview`, while the dev
deployment, which is Vercel's `staging` custom environment rather than a
preview, reports `environment: dev`.

`source` is reserved for the emitter. The events that used it for
something else now use `hire_started.surface`,
`expert_recommended.team_source`, `workflow_installed_on_expert.workflow_source`
and, for the four `first_run_tour_*` events, `opened_from`.

### Identity

- **`distinct_id` is the platform user id** for every backend event, the same
  id the browser passes to `posthog.identify(user.id)`, so both halves land
  on one person.
- **Before login the browser uses the first-party anonymous id**
  (`src/services/analytics/anonymous-id.ts`), passed to `posthog.init` as the
  bootstrap `distinctID`. `identify` merges it into the user at login, and
  `resetAnalyticsIdentity` mints a fresh one at logout.
- **The browser waits for analytics consent** (Cookiebot,
  `src/providers/posthog/posthog-consent.ts`). PostHog starts opted out:
  without consent, browser events are dropped, `identify` is not called and
  the anonymous id lasts only for the page load. Backend events do not
  depend on it.
- **No synthetic ids.** A backend event with no user is dropped
  (`posthog_client.capture` does it for every emitter), since a made-up id
  creates a person nobody can merge.
- **No email or name in event properties.** Identify ids only. The email and
  display name are person properties set by `identify` and nowhere else.
  (`expert_hired.name` is the expert's display name, not a person's.)

### Our own team and impersonation

Our team's clicks and an admin acting as a user would otherwise skew every
chart, funnel and experiment, so both are marked and a project filter leaves
them out (SECRT-2728):

- **`is_internal`** is a person property: true for an admin (Better Auth role
  `admin`) or an `@agpt.co` email, false for everyone else. The backend sets
  it at signup with the other account properties, and again whenever an
  internal account signs in (`POST /api/auth/user`), so someone made an admin
  later is flagged from their next sign-in. `poetry run cli
  posthog-audience-backfill` sets it on earlier accounts. Someone who stops
  being an admin keeps it until that backfill runs again.
- **`impersonated`** is an event property, true on every event an admin
  caused while acting as a user (`X-Act-As-User-Id`). The backend sets it in
  `posthog_client.capture` for every event a request sends with that header
  (`api/middleware/analytics.py`) and every event of a chat turn such an admin
  started; the AutoPilot events above carry it too. The browser stays the
  admin's while impersonating, so its events land on the admin's person; a
  PostHog `before_send` hook adds `impersonated` and `impersonated_user_id`
  to them (`src/services/analytics/posthog-request-context.ts`).
- **An admin acting as a user never changes the user's person.** The backend
  drops `$set_person_properties` while impersonated (the onboarding role, the
  checkout properties). The lifecycle sync still runs, since it mirrors the
  stored trial and Stripe state rather than the admin's action.
- **Session.** With analytics consent, and not while impersonating, the
  browser sends its PostHog session id as `X-PostHog-Session-Id` on API
  requests (`customMutator` through the proxy, and the chat's direct
  requests), and the backend adds it
  as `$session_id` to the events that request sends, so they join the
  browser's session and its replay. Events sent later by the executor, the
  scheduler or a webhook have no session.

The project's "Filter out internal and test users" setting (PostHog project
settings, Product analytics) holds the filter, applied to every insight that
has "Filter out internal and test users" on:

- person property `is_internal` is not `true`;
- event property `impersonated` is not `true`.

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
task: the chat turn that asked for it already counts. `agent_run_started` does fire
for copilot-started runs (with `trigger: copilot`), because it measures
human-initiated runs, not tasks. `user_lifecycle.agent_runs_human_total` is
its SQL twin and includes copilot runs too.

The same definition as a PostHog filter:

```sql
event IN ('agent_run_started', 'chat_message_sent')
AND coalesce(properties.via, '') != 'copilot'
```

In the insight UI: events `agent_run_started` and `chat_message_sent`,
filtered by `via` "is not" `copilot`. Chat turns carry no `via`, so
the filter must keep events where it is unset. The emitters already leave
out dry runs, sub-graph runs, schedule and webhook runs, and automation-origin
turns.

Automated work is measured separately: `schedule_fired` and `trigger_fired`.

## Status legend

| Status | Meaning |
| --- | --- |
| `live` | Sent today under this name. Keep it. |
| `add` | Not sent yet; the name is in `PlannedPostHogEvent` until the change that starts sending it. |

Events that are no longer sent are listed under [Removed](#removed), with
what replaced them.

## Acquisition

| Event | Sender | Status | Required properties | Fires when |
| --- | --- | --- | --- | --- |
| `$pageview` | browser | live | `$current_url` | A route or query string changes (`PostHogPageViewTracker`). |
| `tour_started` | browser | live | — | The public `/tour` page is opened (once per tab). |
| `tour_scenario_started` | browser | live | `scenario` | A tour scenario starts playing. |
| `tour_scenario_completed` | browser | live | `scenario` | A tour scenario reaches its end. |
| `tour_cta_clicked` | browser | live | `label` (`free-trial`, `signup`, `pricing`, `another-scenario`, `self-host`, `share`), `placement` where the CTA has one, `surface` (`tour`, `marketplace`) on sidebar-card clicks | A tour or logged-out marketplace call to action is clicked. |
| `marketing_opted_out` | browser | live | `surface` (`signup`) | Someone refuses marketing email in the app. `signup`: "opt out" is clicked in the legal line under the signup buttons (not "Undo"); nothing else is sent, since there is no user yet, so it only gives the opt-out rate. The refusal itself is stored on the user (`marketingOptOutAt`). |
| `signup_completed` | backend | live | `signup_method` (the auth provider, e.g. `email`, `google`: from the user's first Better Auth account row, where `credential` is reported as `email`, else from a Supabase token's `app_metadata.provider`; omitted when neither has it) | The user row is created (`data/user.py`), whichever request creates it. |

Sidebar CTA clicks retain the tour event names for compatibility. Filter
`surface` to distinguish the marketplace from the tour; older sidebar events
do not carry this property. For tour-only sidebar reports after deployment,
require `surface = tour`. The signup CTA uses `free-trial` in cloud mode and
`signup` on self-hosted installations, where creating an account starts no trial.

The tour funnel also goes to DataFast (`tour_start`, `tour_scenario_start`,
`tour_scenario_complete`, `tour_cta_click`); the PostHog events mirror it with
the same properties and nothing identifying in them: `/tour` is public and
pre-signup, so they land on the visitor's anonymous id. Both need analytics
consent, on `/tour` like everywhere else, but they still count differently:
PostHog holds an event until the visitor's answer arrives, while DataFast drops
any event fired before consent is known. So the DataFast tour funnel misses
mount-time steps (`tour_start`, the first `tour_scenario_start`) on a first
visit where the answer arrives only after the page loads; `tour_start` is
retried on the next tour visit in that session.

## Marketing site

The marketing site, `agpt.co` (repo `Significant-Gravitas/autogpt-marketing-site`),
sends to the same PostHog project as the app, so one funnel runs from a
website visit to a paid account (SECRT-2769). Its code is
`lib/posthog/posthog.ts` and `lib/posthog/goals.ts` there.

- **Consent.** The site asks through Cookiebot too, but Cookiebot keeps one
  answer per domain, so a visitor answers once on `agpt.co` and once on
  `platform.agpt.co`. The site downloads and starts `posthog-js` only once its
  answer allows analytics; before that nothing reaches PostHog, not even
  `/flags`.
- **Settings.** The app's: opted out by default, `localStorage+cookie`
  persistence with consent, autocapture and page leave, no session replay, and
  the same-origin relay at `/relay/pnosthog`.
- **Base properties.** `environment` (`prod`, `preview`, `local`, from
  Vercel), `source: website`, and `revision` (the commit) where the build has
  one.

| Event | Fires when |
| --- | --- |
| `$pageview` | A route or query string changes, and once more when consent arrives, so the landing page counts. |
| `$pageleave`, `$autocapture` | As in the app. |
| `section_viewed` | A section marked for DataFast's scroll goals comes on screen: as soon as any of it shows, and again after it left the screen and comes back. `section` is the part after `scroll_to_` (e.g. `faq`). |
| DataFast goals | Every DataFast goal is also sent to PostHog under an `object_action`, past-tense name, with the same parameters (`data-fast-goal-*` attributes, hyphens turned into underscores, or the properties passed from code, such as `pricing_billing_toggled.billing`). |

The goal names (`POSTHOG_GOAL_EVENTS` on the site, where a test fails for a
goal without one):

| DataFast goal | PostHog event |
| --- | --- |
| `demo_tour_click` | `demo_tour_clicked` |
| `demo_video_play` | `demo_video_played` |
| `demo_video_sound` | `demo_video_sound_toggled` |
| `get_started_click` | `get_started_clicked` |
| `github_selfhost_click` | `github_selfhost_clicked` |
| `github_stars_click` | `github_stars_clicked` |
| `hero_prompt_chip_click` | `hero_prompt_chip_clicked` |
| `hero_prompt_focus` | `hero_prompt_focused` |
| `hero_prompt_submit` | `hero_prompt_submitted` |
| `marketplace_agent_click` | `marketplace_agent_clicked` |
| `marketplace_browse_click` | `marketplace_browse_clicked` |
| `nav_product_click` | `nav_product_clicked` |
| `press_source_click` | `press_source_clicked` |
| `pricing_billing_toggle` | `pricing_billing_toggled` |
| `pricing_click` | `pricing_clicked` |
| `pricing_plan_cta_click` | `pricing_plan_cta_clicked` |
| `product_demo_interact` | `product_demo_interacted` |
| `product_video_play` | `product_video_played` |
| `see_it_in_action_click` | `see_it_in_action_clicked` |
| `signin_click` | `signin_clicked` |
| `testimonial_source_click` | `testimonial_source_clicked` |
| `usecase_cta_click` | `usecase_cta_clicked` |
| `watch_demo_click` | `watch_demo_clicked` |

These names follow the [naming rules](#naming) and are never renamed. They
live in the site's repo, not in `posthog-events.ts`.

### Hand-over to the app

The site keeps PostHog's state in the `ph_<project key>_posthog` cookie on
`.agpt.co`, which `platform.agpt.co` can read. It carries `$device_id`. The
app adopts that id as the visitor's anonymous id, so `identify()` at signup
merges the website pageviews into the new user:

- **While the app's answer is pending,** the app leaves the cookie (and its
  own stored identity) alone and runs PostHog on an in-memory id. Only an
  answer without analytics deletes them, the shared cookie included; the site
  does the same on its side. If Cookiebot asks again after PostHog started
  with an earlier grant, PostHog stops capturing (so it writes nothing until
  the visitor answers) but keeps its storage, the shared cookie included; a
  grant resumes it on the same id.
- **On a grant,** during the page or from an earlier answer, the visitor's
  anonymous id is, in order: the app's own `agpt_anonymous_id`; else the
  device id in the app's own PostHog storage; else the cookie's `$device_id`;
  else the page's in-memory id (`src/services/analytics/anonymous-id.ts`).
  Both device ids are read before PostHog rewrites them. If PostHog started
  on the in-memory id, it moves onto the adopted one without a reload
  (`src/providers/posthog/posthog-consent.ts`).
- **Returning app visitors keep their own id.** A browser that already has
  `agpt_anonymous_id` from an earlier visit with consent keeps it, so
  website visits made under another id do not merge into the account.
- **Only a new visitor or a new account takes the website's id.** The
  cookie's `$device_id` is adopted only when nobody is signed in, or the
  account was created in this browser (the `agpt_account_created` signup flag,
  as seen while signed in on that page: the ads signup conversion clears it).
  An older account signed in on a browser someone else used on agpt.co keeps
  the in-memory id, so `identify()` doesn't merge that person's website visits
  into it. Until the auth store knows the session (after its fallback
  validation; an auth error leaves it unknown for the page), the choice
  waits: PostHog stays opted out and in memory, events
  are held, and nothing is stored. The id then settles, PostHog moves onto it
  and starts, before `identify()`.
- **Logout and account switches forget the shared cookie** when analytics is
  not allowed (declined or still pending): PostHog then runs in memory, so its
  `reset()` would leave the previous person's id there for the next account.
- **The backend gets the id PostHog ends up with.** The attribution report
  (`useReportAttribution`, the `anonymous_id` and `posthog_distinct_id` on
  `UserAttribution`) waits while the answer is pending, then sends the adopted
  id on a grant, or the in-memory id (and no PostHog id) on a decline. It
  goes at once without a Cookiebot domain group, and when uc.js still hasn't
  arrived 10 seconds after the page loaded (blocked). A visitor who never
  answers is reported on the first later page where they do.

## Onboarding and activation

| Event | Sender | Status | Required properties | Fires when |
| --- | --- | --- | --- | --- |
| `onboarding_step_viewed` | browser | live | `step` (`team`, `autopilot`, `role`, `pain_points`, `connect`, `hire`, `preparing`) | A wizard step is shown (once per tab). The paywall step reports `paywall_viewed` instead. |
| `onboarding_completed` | backend | live | — | `ONBOARDING_COMPLETE` is first recorded for the user (`complete_onboarding_step`). |
| `brain_dump_started` | browser | live | `prompt_set` (the role's "Otto might ask" set) | Recording starts. |
| `brain_dump_completed` | browser | live | `input_mode`, `duration_secs` and `finalize_latency_ms` (voice) or `chars` (typed), `prompt_set`, `prompts_seen` (distinct live prompts shown) | A dump was accepted and the wizard advances. `finalize_latency_ms` is the `finalizeBrainDump()` round trip, timed after the upload flush. |
| `brain_dump_canceled` | browser | live | — | Recording is cancelled. |
| `brain_dump_skipped` | browser | live | `prompt_set`, `prompts_seen` | The step is skipped. |
| `brain_dump_recovery_shown` | browser | live | `parts` | A saved partial recording is offered back. |
| `brain_dump_recovery_used` | browser | live | — | The saved recording is used. |
| `brain_dump_retry` | browser | live | `attempt` | A failed finalize is retried. |
| `brain_dump_restarted` | browser | live | — | Recording starts over. |
| `brain_dump_download` | browser | live | — | The recording is downloaded after a failure. |
| `brain_dump_permission_denied` | browser | live | — | Microphone permission is refused, on the first ask and again on each refused retry. |
| `brain_dump_mic_failed` | browser | live | `error_name` | The microphone fails to open, for any reason, a refused permission included. Sent again on each failed retry, so count people, not events. |
| `brain_dump_typed_fallback` | browser | live | `reason` (`chose_to_type`, `permission_denied`), `prompt_set` | The user types instead of speaking. `permission_denied` is sent once per visit, however many retries are refused. |
| `brain_dump_prompt_advanced` | browser | live | `via` (`auto`, `click`), `screen` (`rest`, `typing`, `recording`), `prompt_index`, `prompt_set` | The "Otto might ask" question moves on, by its timer (12 s at rest on narrow screens, 20 s typing, 15 s recording; not while the ideas are paused) or "Another question". Not sent for the first question. |
| `brain_dump_permission_retried` | browser | live | `result` (`granted`, `denied`, `unavailable`), `via` (`retry_link`, `record_instead`, `permission_change`), `permission_state` (`prompt`, `granted`, `denied`, `unknown`) | After a refused microphone, the user asks for it again, or allows the site in its settings (`permission_change`). Sent before `brain_dump_started`. `result` is the browser's answer, not the take: `granted` with no `brain_dump_started` after it means the user had already sent their text, or left the step, while the prompt was open. |
| `transcription_failed` | browser | live | `error_code`, `finalize_latency_ms` | Finalize returns a failure. |
| `welcome_dialog_closed` | browser | live | — | The first-landing intro ends (since SECRT-3149, the first-run walkthrough, finished or skipped). |
| `first_run_tour_shown` | browser | live | `opened_from` (`auto` after onboarding, `menu` from "Show me around") | The guided first run over the real sidebar opens. |
| `first_run_tour_step_viewed` | browser | live | `opened_from`, `step_id`, `step_index`, `target` (what it pointed at once that settled, about half a second: the sidebar item, also inside a phone's opened sidebar sheet; `menu` only where the item is not on screen; or null while the first tasks load) | A walkthrough step is shown, once per step per walkthrough (going Back and Next again does not repeat it). |
| `first_run_tour_completed` | browser | live | `opened_from`, `step_id`, `step_index`, `via` (`start_here`, `connect_tools`) | The last step's "Start here", or its "connect your tools first". |
| `first_run_tour_skipped` | browser | live | `opened_from`, `step_id`, `step_index`, `via` (null) | Skip tour, Close or Escape, at `step_id`. |
| `contextual_tip_shown` | browser | live | `tip` (`second-chat`, `first-cost`, `first-expert`) | A one-time tip appears the first time its surface matters. |
| `contextual_tip_dismissed` | browser | live | `tip` | The tip is put away ("Got it" or Escape); it never shows again. |
| `intro_path` | browser | live | `path` | The intro path (A/B) is chosen. |
| `intro_followup_sent` | browser | live | `chars` | First message after the intro card. |
| `later_dump_completed` | browser | live | — | A brain dump sent later from the composer. |
| `expert_recommended` | browser | live | `template_id`, `position`, `team_source` | A recommended expert card is shown. |
| `expert_recommendation_clicked` | browser | live | `template_id`, `position` | A recommended expert card is clicked. |
| `hire_all_clicked` | browser | live | `count` (experts this click hires), `already_hired`, `recommended`, `retry` | The hire step's primary button is clicked to hire every expert still unhired, or with `retry` the ones a batch failed to hire. |
| `hire_step_continued` | browser | live | `hired`, `recommended`, `via` (`auto` after a full batch, `continue`, `later` for "I'll add members later") | The hire step is left. |
| `tab_intro_shown` | browser | live | `tab` | A tab intro card is shown. |
| `tab_intro_cta_clicked` | browser | live | `tab`, `cta` | Its primary CTA is used. |
| `tab_intro_dismissed` | browser | live | `tab` | It is dismissed any other way. |
| `integration_connected` | backend | live | `provider`, `credential_type`, `method` | A credential is stored (OAuth, key, device code). |
| `credential_oauth_started` | backend | live | `provider` | The backend issues an OAuth login URL (`GET /api/integrations/{provider}/login`). Not Codex. |
| `credential_oauth_exchange_failed` | backend | live | `provider`, `status_code` (unset for an unexpected error), `failure_class` (`invalid_state`, `provider_unavailable`, `token_exchange`, `credential_merge`), `detail` (redacted, at most 200 characters) | `POST /api/integrations/{provider}/callback` returns an error, on any path. Not Codex. |
| `credential_card_never_rendered` | browser | live | `provider`, `failure_class` | A provider is missing from the provider map. |
| `credential_oauth_popup_blocked` | browser | live | `provider`, `failure_class` | Both the popup and the new tab were blocked. |
| `credential_oauth_flow_timed_out` | browser | live | `provider`, `failure_class` | The OAuth flow timed out. |
| `credential_scope_shortfall_blocked_selection` | browser | live | `provider`, `failure_class` | The granted scopes are narrower than required. |
| `credential_proceed_stuck_after_connect` | browser | live | `provider`, `failure_class` | A connected credential never reached the card. |

## Engagement

| Event | Sender | Status | Required properties | Fires when |
| --- | --- | --- | --- | --- |
| `agent_run_started` | backend | live | `graph_id`, `graph_exec_id`, `via` (`manual`, `api`, `copilot`), `via_ref`, `preset_id`; an expert's workflow run adds `expert_id` and `kind: workflow_run` | A person starts an agent run. |
| `chat_message_sent` | backend | live | `chat_session_id`, `origin` (`web`, or the bot platform the chat was opened from: `slack`, `discord`, ...), `kind: chat_turn`, `message_length`; `expert_id` in an expert chat | A person sends a message in an Autopilot or expert chat. |
| `agent_run_finished` | backend | live | `status` (`completed`, `failed`), `graph_id`, `graph_exec_id`, `via`, `expert_id`, `cost_cents`, `duration_seconds`, `is_subgraph_run`; `failure_reason` when failed | A run reaches COMPLETED or FAILED (sub-graph and automated runs included). A top-level expert run is `expert_id` set and `is_subgraph_run` false. Deduplicated on `graph_exec_id`, so a requeue or resume that finishes the same run again sends no second event. |
| `schedule_created` | backend | live | `schedule_id`, `target` (`agent`, `autopilot`, `expert`), `expert_id`, `cron`, `is_recurring`, `run_at`, `graph_id`, `chat_session_id` | Any schedule is registered, from any surface. |
| `chat_tool_called` | backend | live | `chat_session_id`, `tool_name`, `tool_call_id` | The copilot calls a tool. |
| `chat_outcome` | backend | live | `outcome_type` (`agent_run_success`, `schedule_created`), `chat_session_id`; `agent_run_success`: `graph_id`, `execution_id`, `library_agent_id`; `schedule_created`: `target` (`agent`: `graph_id`, `schedule_id`, `cron`, `library_agent_id`; `followup`: `schedule_id`, `target_chat_session_id`, `is_recurring`) | A moment of value in a chat: the copilot started a (non-dry) run or created a schedule for the user. `agent_run_started` / `schedule_created` still count the run or schedule itself. |
| `chat_library_check_outcome` | backend | live | `chat_session_id`, `outcome`, `matches_count`, `top_score` | The create-agent library check ends. |
| `chat_read_screen_shadow_decision` | backend | live | `tool_name`, `tool_call_id`, `source_kind` (`web`, `file`, `shell`, `mcp`, `block`, `workflow`, `chat`, `skill`, …), `chat_session_id`, `trace_id` (the reply's trace, the `$ai_trace_id` of its generations and spans, shared by the subtasks it spawns; the tool call's `$ai_span` with the same `tool_call_id` names the turn), `expert_id`, `verdict` (`allow`, `hold`, `judge_error`), `reason_code`, `judge` (`jev`, `content`; null when the shadow itself failed), `llm_model`, `jev_model`, `rubric_version`, `latency_ms`, `impersonated`, `judgement_id`, `sample_rate` | With `copilot-read-screen` serving `shadow`, the prompt-injection check judges an outside read in a sampled interactive chat (`backend/copilot/gate/shadow.py`). Nothing is held. No content, URL or judge wording is sent. |
| `voice_mode_started` | browser | live | `entry` | Voice mode is switched on. |
| `voice_mode_stopped` | browser | live | `turns`, `state` | Switched off by the user. |
| `voice_mode_timed_out` | browser | live | `turns`, `state` | Closed by the silence timeout. |
| `voice_turn_sent` | browser | live | `turn_index`, `transcript_chars`, `transcribe_latency_ms` | A spoken turn is sent. |
| `voice_turn_dropped` | browser | live | `reason` (`vad_misfire`, `transcribe_failed`, `filler_or_empty`, `interrupted`); `transcribe_latency_ms` with `filler_or_empty`; `turn_index` and `first_sound_latency_ms` (null when nothing played yet) with `interrupted` | A spoken turn is discarded, or with `interrupted` a sent turn's reply is cut off by switching voice mode off or leaving the page. A sent turn ends in exactly one of `voice_turn_completed` or this. |
| `voice_turn_completed` | browser | live | `turn_index`, `first_sound_latency_ms` (null when nothing played) | The mic reopens after the reply. |
| `voice_transcribe_retried` | browser | live | `turn_index` | A failed transcription is retried. |
| `voice_recording_downloaded` | browser | live | `turn_index` | The audio is downloaded instead. |
| `voice_mode_permission_denied` | browser | live | `stage` | Microphone permission is refused. |
| `voice_mode_error` | browser | live | `stage` | The VAD, synthesis or send failed. |
| `experts_section_viewed` | browser | live | — | The marketplace experts shelf renders with results. |
| `expert_profile_opened` | browser | live | `template_id` | An expert profile page opens. |
| `hire_started` | browser | live | `template_id`, `surface` (`onboarding`, `expert_page`) | A hire button is clicked, or once per expert of the hire step's batch. |
| `hire_flow_abandoned` | browser | live | `template_id`, `stage` | The hire dialog is closed before hiring. |
| `expert_hired` | backend | live | `expert_id`, `template_id`, `name`, `failed_preloads_count`, `surface` (`onboarding`, `expert_page`, `copilot`; unset for other API callers) | An expert is hired or an archived one revived, from any surface (`experts_db.hire_expert`). Sent once the hire's background setup settles (`_run_hire_setup`), so `failed_preloads_count` is known; a revival with nothing to set up sends it at once. An idempotent re-hire of an active expert does not fire. |
| `hire_failed` | backend | live | `template_id`, `failed_preloads_count` | Hiring raised. |
| `expert_thread_created` | browser | live | `expert_id` | A new expert chat is created. |
| `writing_style_added` | backend | live | `expert_id` | A writing style is saved on an expert. |
| `workflow_installed_on_expert` | backend | live | `expert_id`, `workflow_source` (`library`, `marketplace`), `library_agent_id` or `store_listing_version_id` | A workflow is attached to an expert. |
| `home_viewed` | browser | live | — | The home dashboard renders with data. |
| `home_attention_actioned` | browser | live | `kind`, `action` | A "needs you" item is approved or declined. |
| `home_team_member_clicked` | browser | live | `expert_id` | A team member row is clicked. |
| `home_suggestion_clicked` | browser | live | `position` (0-based among the rows shown), `surface` (`intro`, `home`), `remaining` (open rows after this one) | A suggested first task is clicked, on the greeting or under the Home composer. |
| `home_suggestion_dismissed` | browser | live | `position` (null for `all`), `surface`, `remaining`, `scope` (`one`, `all`) | A suggested first task is hidden, or "Hide all" is used. An Undo sends nothing. |
| `listing_added_to_library` | backend | live | `store_listing_version_id`, `graph_id`, `library_agent_id` | A marketplace agent is added to the library for the first time. Only a user adding a listing counts (the library route and the add-to-library block). Restoring an entry the user already had does not, and neither does a system install: an add inside another write's transaction, an expert's hire preloads, or attaching a marketplace workflow to an expert (that is `workflow_installed_on_expert`). |
| `listing_downloaded` | backend | live | `store_listing_version_id`, `graph_id` | A signed-in user downloads a marketplace agent file. Signed-out downloads have no user and are not sent. |

## Monetization

| Event | Sender | Status | Required properties | Fires when |
| --- | --- | --- | --- | --- |
| `paywall_viewed` | browser | live | `surface` (`onboarding`, `paywall_gate`, `billing`) | A paywall or plan picker is shown (once per tab per surface): the onboarding subscription step, the in-app paywall modal, the billing page's plan card. The pricing arm comes from PostHog's own `$feature/...` properties. |
| `plan_selected` | browser | live | `subscription_tier` (`PRO`, `MAX`, `BUSINESS` for Team), `billing_cycle`, `surface`; `pricing_variant` on `onboarding`, only when PostHog has assigned an arm (omitted while flags load and for users outside the rollout, who are not in `control`) | A plan is picked: a plan card on either paywall, or Upgrade / Downgrade on the billing page. |
| `billing_portal_opened` | browser | live | `surface` (`billing` for the plan card, `billing_payment_method` for the payment method card) | The Stripe billing portal is opened. |
| `checkout_started` | backend | live | `checkout_kind` (`subscription`, `top_up`, `trial`), `subscription_tier`, `billing_cycle`, `surface` (sent by the client; unset for the low-credit top-up dialog and the legacy `/profile/credits` page), `$feature/<experiment>` for every recorded experiment arm | A Stripe Checkout session is created and its URL returned (subscription, credit top-up or trial card setup). A paid plan changed in place opens no Checkout and sends nothing. |
| `checkout_abandoned` | browser | live | `checkout_kind`, `surface` | The user returns from Stripe Checkout without paying (the cancel URL: `subscription=cancelled`, `topup=cancel`, `trial=cancelled`, or the paywall modal's `paywall_checkout=cancelled` marker). Browser-sent: Stripe only reports the expiry a day later. A trial abandonment is sent once per tab per surface until the next trial checkout starts, so a refresh on the cancel URL does not count twice but a second real abandonment does. |
| `trial_offer_viewed` | browser | live | `trial_offer_version`, `subscription_tier`, `trial_duration_days`, `surface` | A trial offer card is shown. |
| `trial_cancel_popup_viewed` | browser | live | `days_left` (whole days of trial access left, rounded up; `1` under a day) | The "Cancellation confirmed" popup opens, once, right after a cancel that sets the trial to cancel at its end, so trial access continues until then. A reload never reopens it. |
| `trial_resume_clicked` | browser | live | `days_left` | Resume trial is clicked on a trial set to cancel at its end, in the popup or on the trial card. Sent on the click; `trial_resumed` (backend) confirms it took effect. |
| `trial_subscribe_now_clicked` | browser | live | `days_left` | Subscribe now is clicked in the cancellation popup. It opens Billing. There, the trial's own plan converts the trial at once on the saved card (`trial_converted`, no `checkout_started`); any other plan, or a card that asks for authentication, goes through Stripe Checkout (`checkout_started`), which ends the trial without a `trial_ended`. |
| `trial_started` | backend | live | `trial_id`, `trial_offer_version`, `subscription_tier`, `billing_cycle`, `trial_duration_days` | The trial starts. Like every trial lifecycle event, it is sent only when its notification email is queued. |
| `trial_ending` | backend | live | as above | The reminder window opens. |
| `trial_canceled` | backend | live | as above | The trial is set to cancel at its end: the cancellation is scheduled, trial access continues until the end date, and the card is not charged. |
| `trial_resumed` | backend | live | as above | A trial set to cancel is resumed before its end date, so it converts to paid at its end as first agreed. |
| `payment_failed` | backend | live | as above | The conversion charge fails. |
| `trial_converted` | backend | live | as above | The trial converts to paid. |
| `trial_ended` | backend | live | as above | The trial ends without converting. Not sent when the trial subscription ends because another plan was bought while the trial was set to cancel; that plan's `checkout_started` and `payment_succeeded` cover it. |
| `subscription_changed` | backend | live | `change_type: upgrade`, `previous_subscription_tier`, `subscription_tier`, `billing_cycle` | A paid tier upgrade takes effect. Downgrades and cancellations are not sent here yet. |
| `payment_succeeded` | backend | live | `subscription_tier`, `billing_cycle`, `amount_cents` (the invoice's `amount_paid`), `currency` | A subscription invoice is paid. Deduplicated on the invoice id: Stripe delivers both `invoice.payment_succeeded` and `invoice_payment.paid` for one invoice, and redelivers either, so revenue is counted once. |
| `topup_completed` | backend | live | `amount_credits`, `top_up_type`, `amount_cents` (the Checkout total or the auto top-up charge), `currency` | Credits are bought. |
| `subscription_cancellation_scheduled` | backend | live | `subscription_tier` | A paid plan is set to cancel at period end. |
| `subscription_ended` | backend | live | `subscription_tier`, `billing_cycle`, `reason` (Stripe `cancellation_details.reason`: `cancellation_requested`, `payment_failed`, `payment_disputed`; `payment_failed` too when our own payment-failure handler cancelled the subscription because the balance could not cover a failed renewal, which Stripe would stamp `cancellation_requested`) | A paid subscription ends (Stripe `customer.subscription.deleted`), once per subscription, alongside the ended email. A trial that ends unconverted is `trial_ended` instead; a trial ended by buying another plan sends neither. |
| `subscription_tier_reconciled` | backend | live | `direction`, `previous_subscription_tier`, `subscription_tier`, `via` | Ops signal: Stripe and the stored tier disagreed. Not a user action; keep out of funnels. |

The paywall funnel in PostHog is `paywall_viewed` → `plan_selected` →
`checkout_started` → `payment_succeeded`, with `checkout_abandoned`
as the way out.

Browser events captured before PostHog captures (a page's mount effect on a
full page load, such as the return from Stripe, runs before `posthog.init`
and before the consent opt-in) are held by
`src/services/analytics/posthog-capture.ts` while the consent answer is
unknown. With analytics granted (an earlier answer, or a region where
Cookiebot consents on its own once `uc.js` loads) they are sent, with the
time they happened, once PostHog is opted in. When analytics is declined,
there is no consent manager, or the banner is asking, they are dropped: an
answer given on the banner covers what happens after it, not what was held
before it. The once-per-tab guards on `paywall_viewed`, `checkout_abandoned`,
`onboarding_step_viewed` and `tour_started` are set only when the event is
sent or dropped this way, not while it is held.

## Retention

| Event | Sender | Status | Required properties | Fires when |
| --- | --- | --- | --- | --- |
| `schedule_fired` | backend | live | `schedule_id`, `target`, `expert_id`, `graph_id`, `graph_exec_id` or `chat_session_id` | A schedule produces work. |
| `trigger_fired` | backend | live | `webhook_id`, `graph_id`, `graph_exec_id`, `expert_id`, `preset_id`, `target` | A webhook produces a run. |
| `briefing_generated` | backend | live | `run_count`, `decision_count`, `has_content` | A morning briefing is composed (or found empty). |
| `briefing_delivered` | backend | live | `briefing_id` | It is posted to the user's thread. |
| `briefing_opened` | browser | live | — | The briefing renders on home. Not the plan's `briefing_opened_in_chat`, which is opening it in the chat thread. |
| `briefing_outcome_clicked` | browser | live | `status` | An outcome row in the briefing is clicked. |
| `expert_fired` | backend | live | `expert_id` | An expert is fired. |

Return visits are `$pageview`; account-level retention is computed in
`retention_task_weekly` and `user_lifecycle`, not sent as events.

### Email

Postmark's webhooks for the mail the platform sends itself
(`api/features/postmark/`). MailerLite's mail (the onboarding tour, the
changelog) is not covered.

| Event | Sender | Status | Required properties | Fires when |
| --- | --- | --- | --- | --- |
| `email_delivered` | backend | live | `email_type`, `message_stream`, `message_id` | The receiving mail server accepts the email (Postmark's Delivery webhook). |
| `email_opened` | backend | live | `email_type`, `message_stream`, `message_id`, `first_open` | The email's tracking pixel loads (Open webhook). Mail apps that preload images, such as Apple Mail, report opens nobody made, so read it as an upper bound. |
| `email_clicked` | backend | live | `email_type`, `message_stream`, `message_id`, `link` | A link in the email is followed (Click webhook). |
| `email_bounced` | backend | live | `email_type`, `message_stream`, `message_id`, `bounce_type` | The email bounces (Bounce webhook), soft bounces included. A hard bounce also marks the address unverified. |

- **`email_type`** is the Postmark tag the email was sent with: the
  notification type in lower case (`briefing`, `alert`, `verdict`, `ops`,
  `subscription_welcome`, `payment_failed`, `payment_final_notice`,
  `subscription_cancelled`, `subscription_resumed`, `subscription_ended`,
  `trial_update`), or `auth` for Better Auth's verification and password
  emails. Unset for mail sent before tags were added.
- **`message_stream`** is the Postmark stream (`POSTMARK_TRANSACTIONAL_STREAM`)
  and **`message_id`** Postmark's message id, which ties the events of one
  email together.
- **`link`** is the clicked link's scheme, host and path only: the query
  string and fragment are dropped, and a path segment that isn't a plain
  lower-case word, or that looks like an id or a token, becomes `:id`
  (`https://platform.agpt.co/library/agents/:id`). A link that is not
  `http(s)` keeps only its scheme (`mailto:`).
- **`bounce_type`** is Postmark's bounce type (`HardBounce`, `SoftBounce`,
  `Transient`, `SpamNotification`, ...).
- **`first_open`** is whether Postmark counts it as the first open of that
  email.
- **Who it is for.** The recipient address is looked up among users, and the
  user id is the `distinct_id`. An address with no account (such as the Ops
  inbox) sends nothing. The address is never sent or logged.
- **When.** The event carries Postmark's own time, not the time the webhook
  arrived, and is deduplicated on message id, record type and that time (and
  the link, for a click), so a webhook Postmark redelivers counts once.
- **Opens and clicks need tracking on.** Each notification email asks
  Postmark to track opens and to rewrite links in its HTML part only when
  `POSTMARK_TRACK_OPENS_AND_CLICKS` is true (off by default). Auth email is
  never tracked, since its links carry sign-in tokens. Postmark only calls the
  webhook for the record types ticked on the stream's webhook.

## LLM analytics

AutoPilot's LLM calls, tool calls and the ratings of its replies, in
PostHog's LLM analytics (SECRT-2730, SECRT-2979). All are sent server-side
for every user, whatever their cookie consent, like every backend event. The
event names are PostHog's own, not the analytics plan's.

Each AutoPilot reply is one trace, whose id is the turn id of the reply
(`$ai_trace_id`). An expert subtask or sub-session the reply spawns joins the
same trace. No event carries a tool's input or output, or an error message:
they can hold Gmail or Drive content (Google Limited Use).

| Event | Sender | Status | Required properties | Fires when |
| --- | --- | --- | --- | --- |
| `$ai_generation` | OpenRouter Broadcast | live | set by OpenRouter from the request (`backend/copilot/llm_attribution.py`): `distinct_id` = the request's `user` (the platform user id), `$ai_session_id` = its `session_id` (the chat session id), `$ai_span_name` = `trace.generation_name` (`copilot-title`, `copilot-sdk`, `copilot-baseline`), `$ai_trace_id` = `trace.trace_id` (the reply's trace: the id of the executor turn that made the call, or for a subtask the trace of the turn that spawned it, so all calls behind one reply share it; absent for a call made outside a turn), `metadata_environment` (`local`, `dev`, `prod`), `metadata_impersonated`, `metadata_turn_origin` (what started the turn that made the call: `chat`, `scheduled`, `spawned`, `automation` (a workflow's AutoPilot block) or `system`; a reply's own calls match its `$ai_trace` `turn_origin`, a subtask's calls carry the subtask's, as on its `$ai_span`; absent for a call made outside a turn); `metadata_revision` (the deployed build's git sha, `config.sentry_release`) when it is set; model, tokens, cost and latency from OpenRouter | An AutoPilot LLM call goes through OpenRouter: a session title, an SDK turn (main agent, sub-agents and the CLI's compaction, via `CLAUDE_CODE_EXTRA_BODY`) or a baseline turn. |
| `$ai_metric` | backend | live | `$ai_metric_name` (`user-feedback`, `copy`), `$ai_metric_value` (`1` thumbs up or copy, `0` thumbs down), `$ai_session_id` and `chat_session_id` (the chat session id), `message_id` (the reply's row id), `message_sequence`, `feedback_id` (the stored rating, shared with its Langfuse score), `impersonated`; `$ai_trace_id` when the reply row records its trace (`ChatMessage.traceId`, replies written since SECRT-2979); `$ai_feedback_text` when a thumbs-down has a comment | A reply is rated (`POST /api/chat/sessions/{id}/feedback`, `copilot/feedback.py`). Rating a reply again sends another event; the latest per `feedback_id` stands. |
| `$ai_trace` | backend | live | `$ai_trace_id` (the turn id), `$ai_span_name` = `copilot-turn`, `$ai_session_id` and `chat_session_id`, `$ai_latency` (seconds), `$ai_is_error`, `$ai_error` (the error code, when failed), `turn_id`, `outcome` (`completed`, `cancelled`, `failed`), `engine` (`sdk`, `baseline`, ...), `turn_origin` (`chat`, `scheduled`, `automation` for a turn a workflow's AutoPilot block started, `system`; `spawned` on a subtask's span), `tool_call_count`, `impersonated` | A reply's turn ends, if it reached its engine (`copilot/turn_tracing.py`, `copilot/ai_spans.py`). A redelivered turn sends it again under the same event uuid (dedup key = the turn id), so PostHog keeps one; likewise a subtask's span. |
| `$ai_span` | backend | live | `$ai_trace_id`, `$ai_span_id`, `$ai_parent_id`, `$ai_span_name`, `$ai_session_id` and `chat_session_id`, `$ai_latency` (seconds), `$ai_is_error`, `impersonated`. A **tool call**: `$ai_span_name` = the tool's name, `tool_call_id`, `cancelled` (platform tools) or `builtin` (the SDK CLI's own tools, sub-agents' calls included; the high-volume lookups `TodoWrite`, `Read`, `Glob`, `Grep`, `LS` and `NotebookRead` get no span, and no built-in gets one with `CHAT_POSTHOG_BUILTIN_TOOL_SPANS=false`), `$ai_is_error` true when it raised, a gate refused it or it answered with an error, but not when it stopped for an approval card or a sign-in, `$ai_error` = the exception's type name when a platform tool raised (never its message); its parent is the turn (the trace root, or the subtask's span) or the tool that called it. A built-in tool's span is parented to the turn, not to the `Task` span of the sub-agent that called it. A **subtask**: `$ai_span_name` = `copilot-subtask`, `$ai_span_id` = its turn id, parent = the spawning tool's span, which its envelope keeps, so a child that waited in the turn queue or a subtask woken by an approval stays in the trace and under the span its spawner gave it (the trace root for one queued or held before this was recorded), `depth`, plus the `$ai_trace` turn properties, with `turn_origin` = `spawned` (or `scheduled` when the subtask's own turn runs unattended) | A platform tool call ends (`BaseTool.execute`, both engines), an SDK built-in tool call ends (Pre/PostToolUse hooks), or a spawned turn ends. |

What joins them:

- **Person.** A generation's `distinct_id` and a rating's are both the
  platform user id, the id every other event and `posthog.identify` use, so
  both land on the person with their product events and replays.
- **Chat.** `$ai_session_id` is the chat session id on both, so a chat's
  generations and its ratings group in the Sessions tab and join on it.
  `message_id` says which reply was rated.
- **Reply.** `$ai_trace_id` is the reply's turn id on its generations, its
  spans, its `$ai_trace` and its ratings, so they form one trace.
  Generations sit at the trace's top level: Broadcast has no field for a
  generation's parent span, so a subtask's model calls are not nested under
  its span.
- **Environment.** OpenRouter forwards our `trace` keys as `metadata_<key>`,
  so a generation's environment is `metadata_environment`, not the
  `environment` base property; `$ai_metric`, `$ai_trace` and `$ai_span` have
  both base properties as usual. Likewise the build is `metadata_revision` on
  a generation; the backend's events set no build property of their own and
  carry only the base properties our PostHog client adds to every event.
- **Impersonation.** A turn or rating an admin made while impersonating the
  user (`X-Act-As-User-Id`) is still sent, as the user, with
  `metadata_impersonated` / `impersonated` true, so it can be filtered out.
  The mark is kept for a turn queued behind the running-turn cap and for
  sub-sessions such a turn starts. A follow-up sent while a turn runs takes
  that turn's mark, and an engine-switch continuation or an approval wake
  starts unmarked.

Calls that skip OpenRouter never reach Broadcast and have no
`$ai_generation`: direct Anthropic and Claude Code subscription deployments,
a user's connected Codex account and the local transport. Auxiliary calls
that are not attributed yet (the baseline path's context compaction, the
action gate's classifiers, web search, Graphiti memory) arrive without the
user or chat when their client goes through OpenRouter.

## Experiments

| Event | Sender | Status | Required properties | Fires when |
| --- | --- | --- | --- | --- |
| `$feature_flag_called` | browser (posthog-js) | live | set by PostHog | A PostHog flag is read (`useExperiment`). |
| `feature_flag_mismatched` | browser | live | `flag`, `launchdarkly` (`value`, `resolved`), `posthog` (`value`, `resolved`) | Ops signal: with both flag vendors configured, LaunchDarkly and PostHog resolved a flag to different values (`useDualFlag`). Not a user action; keep out of funnels. |

Arms are also stored in the database (`analytics.experiment_assignment`),
so an experiment can be read in PostHog and Looker alike.

## Person properties

The backend keeps these on the person (distinct id = platform user id) with
a `$set` event (`PostHogEvent.SET_PERSON_PROPERTIES`), sent from
`backend/data/posthog_lifecycle_sync.py` after signup, tier changes and every
Stripe subscription sync, and by a daily sweep at 04:15 UTC. The lifecycle
events above mark the moment something happens; these properties hold the
person's current state, so a cohort or breakdown can filter on them without
replaying events.

| Property | Value |
| --- | --- |
| `subscription_status` | One of the values below. Worked out from the current trial and Stripe state, never from the last event received. |
| `signup_at` | When the user row was created. |
| `trial_started_at` | When the trial started. |
| `subscription_started_at` | When the person first became a paying subscriber: the earliest start of any paid subscription, kept across a resubscribe after a gap. For a converted trial, the conversion (its first successful charge), not the trial start. |
| `subscription_canceled_at` | When the current or last paid subscription was canceled, while `subscription_canceled` or `subscription_ended`. |
| `subscription_ended_at` | When it ended, while `subscription_ended`. |

Dates are ISO-8601 UTC strings of the lifecycle moment, not of the sync. A
date that doesn't apply to the current status is `$unset`, never sent as
null.

`subscription_status` counts only **paying** subscribers as `subscribed`
(SECRT-2795). First match wins:

| Value | Meaning |
| --- | --- |
| `subscription_canceled` | A live paid subscription set to cancel; active until the period ends. |
| `subscribed` | A live (`active`) paid subscription. |
| `trial_canceled` | A running trial set to cancel (keeps access until its end, back to `in_trial` on resume), or a trial that ended unpaid because the user canceled it. |
| `in_trial` | A running trial, ours or a Stripe-native one; includes the hour between the trial's end and its first charge. |
| `payment_failed` | The latest paid subscription is `past_due` or `unpaid`, or a trial's first charge failed and Stripe is still retrying it. Access is lost meanwhile. |
| `subscription_ended` | Has paid before, and no paid subscription is live. |
| `trial_expired` | A trial that ended without a payment and without the user canceling it: it ran out, or its first charge never cleared and Stripe gave up. |
| `signed` | Anything else, including a signup that never started a trial, and a former payer now on a free (fully discounted) subscription, who keeps `subscription_started_at`. |

- **Paid** means the subscription charges something. A subscription whose
  discounts take the whole price (a 100% coupon, or an amount off at least the
  price, on the subscription or on the customer) is not paid: it never makes
  anyone `subscribed`, `subscription_canceled` or `subscription_ended`, and
  its start is not a paid start. Neither is one whose only invoice so far
  charged nothing (a 100% coupon for the first period, which Stripe takes off
  the subscription once that invoice is made). A partial discount is paid. A
  full discount that ends (a repeating coupon) makes the subscription paid
  from then on. Stacked discounts apply one after another, in Stripe's
  order. A customer's discount counts only from its start, so a coupon given
  later doesn't make an older subscription free. Someone who paid before and
  now has a live free subscription (another one, or the same one given a
  coupon after it first charged) is `signed` (not `payment_failed` or
  `subscription_ended`), keeping their first paid start.
- **The plan tier is not read.** An admin-granted plan (ENTERPRISE, BUSINESS
  or any other) with no paid Stripe subscription is `signed`; one who pays
  through Stripe is `subscribed` by the usual rule.
- **The user canceled** a trial when it was set to cancel, or canceled on
  request without a comment of ours (a comment the customer typed in the
  billing portal still counts as theirs). Our own cancels (after a failed
  payment, of a rejected trial, of a replaced plan) carry a comment starting
  `autogpt:` or `autogpt_trial:`, and Stripe giving up on the first charge
  says `payment_failed`: both are `trial_expired`.

MailerLite's `subscription_status` and its `*_date` fields hold the same
values by the same rules: both come from one mapping
(`backend/data/posthog_lifecycle.py`). Each lifecycle sync queues them for
MailerLite as well (`backend/notifications/lifecycle_audience.py`), written
only into a subscriber MailerLite already has and only when they differ. The
daily sweep is PostHog only, so nothing corrects MailerLite on a schedule: a
sync that failed before it was queued, or two syncs from different pods queued
out of order, leave it stale until the user's next sync. `poetry run cli
mailerlite-backfill --fields-only --apply --yes` (after its dry run) catches
MailerLite up. MailerLite holds days, not moments.

The role picked in the onboarding wizard is set with a `$set` of its own when
the profile is submitted at the Preparing step (`POST /api/onboarding/profile`,
`product_analytics.set_onboarding_role`), so funnels and retention can be split
by role. Someone who left the wizard before Preparing has neither property.

| Property | Value |
| --- | --- |
| `onboarding_role` | The pick, as labelled in the wizard: `Founder / CEO`, `Operations`, `Sales / BD`, `Marketing`, `Product / PM`, `Engineering`, `HR / People` or `Other`. |
| `onboarding_role_other` | What was typed after picking `Other` (trimmed, at most 100 characters); null for any other pick. |

The pick is kept on `UserOnboarding`, not read back from the AutoPilot
business understanding, whose copy AutoPilot rewrites. `poetry run cli
onboarding-role-backfill` sets both for earlier accounts with `$set_once`, from
exact picks only: any other stored value may be `Other`'s text or a rewrite,
so it is left unset. MailerLite gets the same values as `role` and
`role_other`.

The fields GTM segments checkout openers on in MailerLite are on the person
too (SECRT-2895, `backend/notifications/posthog_audience.py`). They are worked
out by the same code (`audience_enrichment.checkout_fields`) and named as in
MailerLite, except the checkout open, which is a moment here and a day there.
Every account gets `email_type`, `signin_method` and `is_internal` (PostHog
only) at signup. Every checkout opener gets the MailerLite fields from the
checkout routes and the completed-checkout webhook. Unlike
MailerLite, PostHog also gets accounts that opted out of marketing or that a
signal places in Iran or Russia: it is analytics, not marketing.

| Property | Value |
| --- | --- |
| `email_type` | `business`, `personal` (a webmail provider) or `education`. |
| `is_internal` | `true` for our own team (an admin or an `@agpt.co` email), else `false`; see [Our own team and impersonation](#our-own-team-and-impersonation). Every account, not only openers. |
| `signin_method` | `google` or `email`; `google` wins for an account that has both. |
| `checkout_opened_at` | When the account first opened Stripe checkout. |
| `country` | The country's English name, from the strongest source seen (see `country_source`). |
| `country_code` | The same country as an ISO 3166-1 alpha-2 code. |
| `country_source` | `stripe` (the billing address), `ip` (the visitor's IP country at checkout) or `timezone` (the browser's). |
| `exclude_de_at` | `yes` when any signal points at Germany or Austria, else `no`. |

MailerLite reads the subscriber before each write so it never makes a value
worse; PostHog can't be read on the way, so `$set_once`, which only fills in
what the person doesn't hold yet, keeps the same rules. The first open stays.
`exclude_de_at` `yes` is set and `no` only filled in, so it never goes back to
`no`. A billing country is set, so it always wins, and an IP or timezone
country is only filled in, so it never replaces one. The one difference:
PostHog keeps the first IP or timezone country it gets, where MailerLite moves
on to a later IP country. `poetry run cli
posthog-audience-backfill` sends them for earlier accounts, by the same rules,
from Stripe's first Checkout Session (set outright, since Stripe holds every
open) and its billing country or else the browser's timezone. An IP country is
never backfilled, since it was never stored.

## Removed

No longer sent (SECRT-2722, SECRT-2723). The names stay reserved: the pin tests fail if
one comes back, because reusing it would splice a different action onto the
history PostHog already holds.

| Event | Was sent by | Read instead |
| --- | --- | --- |
| `subscription_trial_checkout_started` | browser | `checkout_started` with `checkout_kind: trial`, sent by the backend once the trial's Checkout session exists; `surface` is the same `onboarding` / `billing`. `trial_offer_version` is dropped; `trial_started` carries it. |
| `hire_completed` | backend | `expert_hired` (same hire; it now also skips idempotent re-hires). |
| `hire_flow_completed` | browser | `expert_hired` with `surface: expert_page`. `elapsed_ms` is PostHog's time to convert from `hire_started`; `voice_picked` is dropped. |
| `onboarding_expert_hired` | browser | `expert_hired` with `surface: onboarding`. The card `position` is on `expert_recommendation_clicked`. |
| `copilot_message_sent` | backend | `chat_message_sent`, which carries `message_length`. |
| `expert_run_completed` | backend | `agent_run_finished` with `expert_id` set and `is_subgraph_run: false`. |
| `finalize_latency_ms` | browser | The `finalize_latency_ms` property on `brain_dump_completed` and `transcription_failed`. |
| `voice_transcribe_latency_ms` | browser | The `transcribe_latency_ms` property on `voice_turn_sent` (and `voice_turn_dropped` with `reason: filler_or_empty`). |
| `voice_first_sound_latency_ms` | browser | The `first_sound_latency_ms` property on `voice_turn_completed`, or on `voice_turn_dropped` with `reason: interrupted` for a reply that was cut off. |
| `experiment_exposed` | browser | Nothing: `useLaunchDarklyExperiment` had no caller and was deleted. PostHog-bucketed experiments use `$feature_flag_called`. |
| `copilot_trigger_setup` | backend | Nothing: never sent (no caller). |
| `intro_card_dismissed`, `raise_door_clicked`, `intro_start_with_autopilot` | browser | Nothing: declared, never sent. |
| `capability_card_viewed`, `capability_cards_completed`, `capability_cards_skipped` | browser | `first_run_tour_step_viewed`, `first_run_tour_completed` and `first_run_tour_skipped` (SECRT-3149): the four concept cards were replaced by the walkthrough over the real UI. |

## Renamed

Renamed once to the analytics plan's names (SECRT-2722). The old names are
reserved like removed ones; an insight on the old name must be pointed at the
new one.

| Old name | New name | Filter to keep the old meaning |
| --- | --- | --- |
| `run_agent` | `agent_run_started` | `expert_id` is not set |
| `run_expert` (`kind: workflow_run`) | `agent_run_started` | `expert_id` is set |
| `run_autopilot` | `chat_message_sent` | `expert_id` is not set |
| `run_expert` (`kind: chat_turn`) | `chat_message_sent` | `expert_id` is set |
| `agent_run_completed` | `agent_run_finished` | `status = completed` |
| `agent_run_failed` | `agent_run_finished` | `status = failed` |
| `copilot_tool_called` | `chat_tool_called` | — |
| `copilot_library_check_outcome` | `chat_library_check_outcome` | — |
| `copilot_agent_run_success` | `chat_outcome` | `outcome_type = agent_run_success`. Dry runs no longer count; `graph_name` is dropped. |
| `copilot_agent_scheduled` | `chat_outcome` | `outcome_type = schedule_created` and `target = agent`. `graph_name` and `schedule_name` are dropped. |
| `copilot_followup_scheduled` | `chat_outcome` | `outcome_type = schedule_created` and `target = followup`. `chat_session_id` is now the chat that scheduled it; the destination is `target_chat_session_id`. |
| `subscription_trial_offer_viewed` | `trial_offer_viewed` | — |
| `subscription_trial_started` | `trial_started` | — |
| `subscription_trial_ending` | `trial_ending` | — |
| `subscription_trial_canceled` | `trial_canceled` | — |
| `subscription_trial_resumed` | `trial_resumed` | — |
| `subscription_trial_converted` | `trial_converted` | — |
| `subscription_trial_ended` | `trial_ended` | — |
| `subscription_trial_payment_failed` | `payment_failed` | — (only trial conversion charges send it today) |
| `subscription_upgraded` | `subscription_changed` | `change_type = upgrade` |
| `subscription_payment_success` | `payment_succeeded` | — |
| `credit_topup_success` | `topup_completed` | — |
| `subscription_tier_reconciliation_discrepancy` | `subscription_tier_reconciled` | — |
| `feature_flag_mismatch` | `feature_flag_mismatched` | — (not in the analytics plan; renamed to the `object_action` past-tense form before it reached production) |

### Renamed properties

Renamed in the same release as the events, to the analytics plan's envelope,
so filters and breakdowns break once. Values are unchanged unless noted.

| Old property | New property | On |
| --- | --- | --- |
| `session_id` | `chat_session_id` | `chat_message_sent`, `chat_tool_called`, `chat_outcome`, `chat_library_check_outcome`, `schedule_created`, `schedule_fired` |
| `target_session_id` | `target_chat_session_id` | `chat_outcome` (`target = followup`) |
| `trigger` | `via` | `agent_run_started`, `agent_run_finished` |
| `trigger_ref` | `via_ref` | `agent_run_started` |
| `surface` (`chat`, or the bot platform) | `origin` (`web`, or the bot platform) | `chat_message_sent`. The plan's `origin` is the channel. |
| `origin` (`interactive`) | removed | `chat_message_sent`. Automation turns are not sent, so it was always `interactive`. |

## Differences from the analytics plan

Left for follow-up changes, so this list and the plan can be compared line by
line:

- **Events the plan folds into others keep their names until that change.**
  The brain-dump events (`brain_dump_*`, `transcription_failed`,
  `later_dump_completed`) and the wizard's `intro_path` and
  `hire_step_continued` become properties of `onboarding_step_viewed` /
  `_completed` / `_skipped` / `_back`. Spoken turns (`voice_turn_sent`)
  become `chat_message_sent` with `input_mode: voice`.
- **`subscription_changed` covers upgrades only.** The plan also counts
  cancellations and downgrades there; `subscription_cancellation_scheduled`
  and `subscription_ended` stay separate events.
- **`chat_outcome` has two outcome types.** Only `agent_run_success` and
  `schedule_created` have an emitter. `agent_created`, `trigger_setup`,
  `artifact_created`, `file_produced` and `answer_only` do not.
- **Plan events we do not send yet** (`signup_started`, `chat_session_started`,
  `chat_response_completed`, `chat_blocked`, `screen_viewed`,
  `screen_engaged`, `milestone_reached`, the builder and library families,
  `email_sent`, ...) come with the plan's phases, not with this list.
- **`email_delivered` instead of the plan's `email_sent`.** Postmark's
  webhook reports that the receiving server accepted the email, which is
  delivery, not sending. `email_sent` stays free for the send itself.
- **Events the plan has no name for keep their own**, e.g.
  `integration_connected`, `schedule_created`, `hire_started`,
  `billing_portal_opened`, `marketing_opted_out`, `tour_*`,
  `tab_intro_*`, `voice_*` and `credential_*`. `briefing_opened` is the
  briefing shown on home, a different action from the plan's
  `briefing_opened_in_chat`.

## Events not in the constants modules

These are sent by PostHog or a third party, not by our code, and are listed
so nobody adds a duplicate:

- `$pageleave`, `$autocapture`, `$identify` and `$feature_flag_called`:
  posthog-js (`capture_pageleave` and `autocapture` are on).
- `$ai_generation`: OpenRouter Broadcast, for every AutoPilot LLM call that
  goes through OpenRouter (see [LLM analytics](#llm-analytics)).
