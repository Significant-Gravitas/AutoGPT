# Feature flags

Which vendor answers a flag read is set per deployment by `FEATURE_FLAG_BACKEND`:
`launchdarkly` (the default), `posthog`, or `dual` — evaluate both, serve
LaunchDarkly's answer, and log every disagreement. Call sites are unaffected by
the choice; they read through `evaluate_feature_flag()` on the backend and
`useGetFlag` / `useFlagStatus` on the frontend.

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
`POSTHOG_PERSONAL_API_KEY` is set — without that key the SDK evaluates over the
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
so it keeps the job — and because the lock expires after two poll intervals, a
refresher that dies hands the job to another process within about a minute. A
process shutting down cleanly releases the lock immediately.

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
| `refresher` / `stored` | this process holds the lock and wrote the definitions |
| `follower` / `cached` | this process read the shared copy |
| `stale` | the shared copy is older than five poll intervals and is being served anyway |
| `empty` | nothing in the cache; this process fetches from PostHog directly |
| `error` | Redis was unreachable |

Healthy, with N replicas: `stored` increments on exactly one process, `cached`
on the other N−1, and `error` stays flat.

### Failure modes

| What happens | What the platform does |
|---|---|
| Cache empty at boot | The process fetches from PostHog directly, once, and the refresher fills the cache. |
| Redis unreachable | Every process polls PostHog for itself, as it would without the cache, and logs once per five minutes. Flag reads are unaffected. |
| Shared copy goes stale | It is served anyway, with a warning, while the refresher catches up. Stale definitions beat none. |
| The refresher dies | Its lock expires and another process takes the job on its next poll. |
| A definitions fetch fails | The definitions already in memory stay in use, and the cache keeps the last good copy. |

Nothing here can make a flag read wait on Redis or on PostHog: flag evaluation
only ever reads the definitions already in the process.
