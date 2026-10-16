# Feature flags

Which vendor answers a flag read is set per deployment by `FEATURE_FLAG_BACKEND`:
`launchdarkly` (the default), `posthog`, or `dual` — evaluate both, serve
LaunchDarkly's answer, and log every disagreement. Call sites are unaffected by
the choice; they read through `evaluate_feature_flag()` on the backend and
`useGetFlag` / `useFlagStatus` on the frontend.

## One set of flags, separate reporting

Flags are defined once, in the **prod** PostHog project. Dev reads those
definitions, while every event and person they produce
goes to the **dev** project, so prod's analytics never mix with dev's.

Only the backend reads flag definitions. The browser gets its flags from the
backend's `GET /api/feature-flags` (`NEXT_PUBLIC_FEATURE_FLAG_BACKEND=server`),
which evaluates them through the same path as the backend's own reads. So the
split below covers the browser's flags too, with no PostHog flags client in
the browser.

> **`server` mode arrives with #15383.** Until that PR is merged, the browser's
> backend selector (`frontend/src/services/feature-flags/flag-backend.ts`)
> knows only `launchdarkly`, `posthog` and `dual`, and treats any other value,
> `server` included, as `launchdarkly`. Set `server` on dev only once #15383 is
> deployed there; before that, the split covers the backend's reads only.

| Setting | Prod | Dev |
|---|---|---|
| `POSTHOG_API_KEY` (backend) | prod project API key | dev project API key |
| `POSTHOG_FLAGS_API_KEY` (backend) | blank | prod project API key |
| `POSTHOG_SECRET_KEY` (backend) | prod Feature Flags Secure API Key | prod Feature Flags Secure API Key |
| `NEXT_PUBLIC_POSTHOG_KEY` (Vercel) | prod project API key | dev project API key |

This applies to flags PostHog answers, so dev needs `FEATURE_FLAG_BACKEND=posthog`
(under `dual`, LaunchDarkly's answer is served and PostHog's is only compared),
and `NEXT_PUBLIC_FEATURE_FLAG_BACKEND=server` (once #15383 is deployed) so the
browser asks the backend.

A blank flags key reads flags from the analytics project, which is what prod
and every self-hosted deployment do. `POSTHOG_SECRET_KEY` always belongs to the
project flags are read from. All projects must be in the one region
`POSTHOG_HOST` names.

**Backend.** The flag client uses the flags key and the secret key, so it polls
prod's definitions and evaluates in-process. When the flags key names another
project it is built with `send=False`, so its `$feature_flag_called` events
never leave the process, and every read is local only
(`only_evaluate_locally`): a flag local evaluation cannot answer serves its
default rather than asking prod's `/flags` with the user's email. That makes
the secret key required on dev; without it every read serves its default, and
the client logs a warning at startup. The analytics client
(`backend/util/posthog_client.py`) keeps `POSTHOG_API_KEY`.

This covers every PostHog flag read the backend makes: its own
(`evaluate_feature_flag`, `get_feature_flag_value`, route gates) and the
browser's through `/api/feature-flags`.

**Browser.** In `server` mode the browser asks the backend and nothing else
for its flags, so on dev it gets prod's definitions with no change of its own.
posthog-js still talks only to the analytics project. The other
`NEXT_PUBLIC_FEATURE_FLAG_BACKEND` values read flags in the browser itself:
`launchdarkly` (the default) from LaunchDarkly, and `posthog` / `dual` from the
project `NEXT_PUBLIC_POSTHOG_KEY` names. So dev must run `server` mode to share
prod's flags. The one browser read left is `useExperiment`, which reads posthog-js and
so, on dev, the dev project's experiment flags, until experiment arms are
assigned on the backend.

**Rolling out to dev only.** Every PostHog flag read the backend makes, its own
and the browser's in `server` mode, carries the person property `environment`,
the backend's `APP_ENV`: `local`, `dev` or `prod`, the same value the analytics
events carry. A release condition `environment = dev` turns a flag on for dev
alone; prod never matches it. A Vercel preview in `server` mode gets the answer
of the backend it calls. The property is only an evaluation input: it is never
stored on a person.

Browser reads through posthog-js (`posthog` / `dual` mode, which is prod today)
don't carry `environment`: they send `buildFlagPersonProperties` (email, email
domain, role, created at) only. So use `environment` only in positive
dev-only conditions like `environment = dev`. A condition like
`environment = prod` or "`environment` is not `dev`" would turn on in prod's
backend reads but stay off in prod's browser until prod runs `server` mode.

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
development and CI run `FEATURE_FLAG_BACKEND=launchdarkly`, so none of this is
reached; set `POSTHOG_FLAG_DEFINITION_CACHE=memory` if you want to exercise
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

Nothing here can make a flag read wait on Redis or on PostHog: flag evaluation
only ever reads the definitions already in the process.
