# Feature flags

Which vendor answers a flag read is set per deployment by `FEATURE_FLAG_BACKEND`:
`posthog` (the default), `launchdarkly`, or `dual` — evaluate both, serve
LaunchDarkly's answer, and log every disagreement. Without `POSTHOG_API_KEY`
the default reads every flag as its code default, which is what a self-hosted
install and CI get. Call sites are unaffected by
the choice; they read through `evaluate_feature_flag()` on the backend and
`useGetFlag` / `useFlagStatus` on the frontend.

## Product defaults and operator overrides

The backend's `util/feature_flag/defaults.py` enables these mature boolean
flags when no vendor answer is available: `autogpt-new-layout`, `chat-search`,
`artifacts-page`, `hire-experts`, `skills-hub`, `onboarding-brain-dump`,
`onboarding-expert-team`, `graphiti-memory`, `graphiti-communities-enabled`,
`expert-spend-approval`, and `ai-agent-execution-summary`. The frontend's
`defaultFlags` map matches the flags it consumes. The marketplace search-term
payload defaults to `["Research", "Marketing", "Product", "SEO", "Business"]`.

Omitting a boolean helper's `default` argument (or passing `None`) selects the
per-flag default; an explicit `default=False` or `default=True` overrides only
the fallback. A vendor's boolean answer and `FORCE_FLAG_<NAME>` /
`NEXT_PUBLIC_FORCE_FLAG_<NAME>` overrides still win, including explicit false.
Unlisted flags still default false unless their caller supplies another
fallback. Payment, hosted Stripe/trial payloads, the recent stream runtime,
and experimental dream gates are unchanged.

A fallback is not authoritative. The backend's frontend endpoint omits
unanswered values so the browser applies its matching defaults. Memory
registration may use the community fallback to register work, but an
unavailable flag cannot erase an existing schedule's recorded intent.
Unauthoritative reads are reported as `flag_unavailable`, separately from
registration failures; actual off answers retain the existing cleanup
behavior. User, authentication, organization, entitlement, and activity
prerequisites still apply. Spend approvals do not run with `ENABLE_CREDIT=false`.

Memory and community defaults require the existing FalkorDB and model
configuration; community maintenance, briefings, and execution summaries can
perform background model work. Operators can disable their individual flags
without enabling payments or any experimental dream stage.

PostHog's single-flag SDK result conflates inactive or missing keys with failed
reads. Backend reads of the selected mature booleans therefore try loaded local
definitions, then at most one
remote decision request. A complete response (`errorsWhileComputingFlags=false`)
that omits the key means off; an incomplete or failed response omitting the key
uses the fallback without authority. Valid per-key answers, including false,
still win in partial responses. Unselected flags, payloads, and experiment reads
retain their existing path and fallback semantics.
In direct browser PostHog mode, the public subscription callback drops partial
evaluation metadata. A scoped adapter at the SDK's typed `receivedFeatureFlags`
boundary preserves that metadata before subscribers run and restores the receiver
when tracking ends. A complete response omitting a mature boolean answers false;
partial or unknown completeness keeps `answered=false` for that omission. HTTP failures
retain usable cached values; when none is available, or the provider is
unconfigured, the selected fallback applies. The backend-served mode
remains the default.

Use the [Tuesday audit prompt](feature-flag-audit-prompt.md) to evaluate later
default changes against rollout history and actual consumer behavior.

## One PostHog project per environment

Dev and Prod each have their own PostHog project, and each project holds its
own flags. A deployment reads its flags from the same project its analytics go
to, the one `POSTHOG_API_KEY` names, and evaluates them locally with that
project's Feature Flags Secure API Key in `POSTHOG_SECRET_KEY`. Where local
evaluation can't decide a read, the SDK asks PostHog's `/flags` endpoint
instead, unless the read is local-only, as a signed-out visitor's own read is.

| | Dev | Prod |
|---|---|---|
| PostHog project | [281733](https://eu.posthog.com/project/281733/feature_flags) | [117338](https://eu.posthog.com/project/117338/feature_flags) |
| `POSTHOG_API_KEY`, `POSTHOG_SECRET_KEY` (backend) | Dev project's | Prod project's |
| `NEXT_PUBLIC_POSTHOG_KEY` (Vercel) | Dev project's | Prod project's |

PostHog has no environments within a project, so a flag both deployments read
is defined in each project. The **Projects** tab on the
[flag list](https://eu.posthog.com/project/281733/feature_flags?tab=projects)
and on each flag's page shows a flag across both projects side by side.

Dev and Prod both run `FEATURE_FLAG_BACKEND=posthog`, so a flag is changed in
that environment's PostHog project and LaunchDarkly answers nothing there.

**The browser.** With `NEXT_PUBLIC_FEATURE_FLAG_BACKEND=server`, the default,
the browser gets its flags from the backend's `GET /api/feature-flags`, which evaluates
them through the same path as the backend's own reads, so the browser and the
backend give one answer from one project. A Vercel preview in `server` mode
gets the answer of the backend it calls, and a backend with no vendor answers
`configured: false`, so the browser serves its own defaults and stops asking.
The other values read flags in the browser itself: `launchdarkly` from
LaunchDarkly, and `posthog` /
`dual` from the project `NEXT_PUBLIC_POSTHOG_KEY` names. `useExperiment` reads
posthog-js in every mode.

**The `environment` property.** Every PostHog flag read the backend makes, its
own and the browser's in `server` mode, carries the person property
`environment`: the backend's `APP_ENV` (`local`, `dev` or `prod`), the same
value its analytics events carry. A release condition can use it in either
project like any other person property. It is only an evaluation input, never
stored on a person. Browser reads through posthog-js (`posthog` / `dual` mode)
don't carry it: they send `buildFlagPersonProperties` (email, email domain,
role, created at) only, so a condition on `environment` matches only the
backend's reads there.

**Splitting signed-out visitors.** A visitor who allowed analytics is bucketed
under `visitor:<their PostHog id>`, but only by conditions their read can
decide: a rollout percentage, `environment` and `is_anonymous`, with no email.
A condition group on another person property or a cohort (the migrated
"email in employees", for one) can't be decided for a visitor, so a visitor
whom no decidable group matches gets the shared signed-out answer instead.
"Employees OR 20%" is then all or nothing for visitors, whichever side the
shared `anonymous` key lands on. For a real 20% of visitors, add
`is_anonymous` is not `true` to the rollout group: the shared answer is then
off, and only each visitor's own bucket turns the flag on.

## Sharing PostHog's flag definitions

PostHog evaluates flags locally, in the process, from a set of flag definitions
it polls for every 30 seconds. PostHog
[bills one definitions fetch as ten flag requests](https://posthog.com/docs/feature-flags/local-evaluation),
so a poller in every process makes that bill scale with replica count: at two
polls a minute, each process costs 28,800 flag requests a day whether or not it
reads a single flag.

So one process is elected to refresh the definitions and write them to Redis,
and every other process evaluates from that shared copy. The fleet then costs
one process's worth of polling — 28,800 flag requests a day in total, instead of
28,800 per replica.

This is PostHog's own `FlagDefinitionCacheProvider` interface; the SDK calls it
from its polling thread, never from a flag read, so no request can block on
Redis.

### Settings

| Setting | Default | What it does |
|---|---|---|
| `POSTHOG_FLAG_DEFINITION_CACHE` | `redis` | `redis` shares definitions through Redis; `memory` keeps them in the process; `none` restores one poller per process. |
| `POSTHOG_FLAG_DEFINITION_REFRESH_SECONDS` | `30` | How often the refresher fetches, and how often everyone else re-reads the shared copy. |
| `POSTHOG_FLAG_DEFINITION_CACHE_TTL_SECONDS` | `600` | How long the shared copy stays readable after the last refresh. |

None of them is read unless `FEATURE_FLAG_BACKEND` is `posthog` or `dual` **and**
`POSTHOG_SECRET_KEY` is set — without that key the SDK evaluates over the
wire and has no definitions to share.

**Per environment.** Leave the defaults on `dev` and `prod`: both run several
replicas against a shared Redis, which is what the cache is for. Local
development and CI have no PostHog keys, so none of this is reached; set `POSTHOG_FLAG_DEFINITION_CACHE=memory` if you want to exercise
PostHog locally without Redis.

### What the refresher is

Every process tries `SET <key> <id> NX PX` on one Redis key each time it polls.
The one that gets it fetches the definitions from PostHog and writes them to
Redis; the others read that copy. The winner renews its own lock on every poll,
so it keeps the job. The lock expires two poll intervals after its last renewal,
and the next process to poll takes it, so a refresher that dies is replaced
within 90 seconds at the defaults. One shutting down cleanly releases the lock
immediately, and is replaced on the next poll.

### Telling it is working

One process logs, once, when it takes the job:

```
This process now refreshes the shared PostHog flag definitions (<host>:<pid>:<id>)
```

and the others log `Another process now refreshes the shared PostHog flag
definitions` when they lose it.

The counter `autogpt_posthog_flag_definition_cache_events_total` on `/metrics`
carries the same story, by `outcome`:

| outcome | meaning |
|---|---|
| `refresher` / `stored` | this process holds the lock and wrote its definitions to the shared copy, which it does every poll whether or not PostHog reported a change |
| `follower` / `cached` | this process read the shared copy |
| `stale` | the shared copy has not been rewritten for five poll intervals, so no refresher is writing it; it is served anyway |
| `empty` | nothing in the cache; a process with no definitions yet fetches from PostHog directly, and one that has some keeps them |
| `error` | Redis was unreachable |

Healthy, with N replicas: `stored` increments once per poll on exactly one
process, `cached` once per poll on each of the other N−1, and `empty` and
`error` stay flat.

### Failure modes

| What happens | What the platform does |
|---|---|
| Cache empty at boot | The process fetches from PostHog directly, once, and the refresher fills the cache. |
| Redis unreachable | Every process polls PostHog for itself, as it would without the cache, and logs once per five minutes. Flag reads are unaffected. |
| Shared copy goes stale | It is served anyway, with a warning, while the refresher catches up. Stale definitions beat none. |
| The refresher dies | Its lock expires and another process takes the job on its next poll. |
| A definitions fetch fails | The definitions already in memory stay in use, and the refresher keeps writing that copy to the cache. |

Nothing here can make a flag read wait on Redis: evaluation reads the
definitions already in the process, and only a read they can't decide waits on
PostHog's `/flags`.
