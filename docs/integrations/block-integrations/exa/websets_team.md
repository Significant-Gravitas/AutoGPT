# Exa Websets Team
<!-- MANUAL: file_description -->
Blocks for checking your team's Exa Websets capacity.
<!-- END MANUAL -->

## Exa Get Websets Team Info

### What it is
Get your team's Websets concurrency usage and limits, to check capacity before launching large websets

### How it works
<!-- MANUAL: how_it_works -->
This block reads the Websets API's `/v0/teams/me` and returns how many requests are active and queued against the team's concurrency and queue limits. An empty limit means unlimited.

`available_slots` is max concurrent minus active (never below 0), and `has_capacity` is true when a slot is free or there is no limit; it does not consider the queue. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| team_id | The team's unique identifier | str |
| name | The team's name | str |
| active | Websets requests being processed right now | int |
| queued | Websets requests waiting to start | int |
| max_concurrent | Most requests the team can run at once (empty means unlimited) | int |
| max_queued | Most requests the team can queue (empty means unlimited) | int |
| available_slots | Requests that can start immediately: max concurrent minus active (empty means unlimited) | int |
| has_capacity | True if a new request would start now instead of queueing | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Pre-Launch Check**: Confirm there is a free slot before creating a large webset.

**Backoff Logic**: Wait and retry later when has_capacity is false instead of adding to the queue.

**Usage Monitoring**: Track active and queued requests over time to see whether the plan's limits are enough.
<!-- END MANUAL -->

---
