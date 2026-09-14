---
name: weekly-homelab-sweep
description: Produce a short read-only weekly report on a self-hosted setup - which updates actually matter, what changed about exposure, and whether backups are fresh and tested. Use for a weekly homelab digest, a maintenance check-in, or when asked what needs doing on the servers.
triggers:
  - weekly sweep
  - what needs updating
  - homelab check
  - anything i should do on the servers
version: "1"
---

# Weekly homelab sweep

One short report, read in under a minute, answering: is anything unsafe,
and what is worth my Saturday. Everything else is a count.

This skill is **read-only**. It never connects to a host, never runs a
command, never changes anything. That is what makes it safe on a schedule.

## This usually runs unattended

Most runs have nobody watching, so **never make a question the whole
output**. If you cannot see the setup, say so as the report and name what
you need once:

> No host inventory is connected, so I can't sweep. Paste your compose
> files or connect a source and I'll run properly from next week.

Then stop. A scheduled job whose entire output is "which server?" delivers
nothing, every week, silently. If the reader supplied the state in their
message, report from that and label it as reported rather than verified.

## Gather

Since the last sweep (default: 7 days):

1. **Security updates** — versions behind where the gap includes a fix for
   a known vulnerability. This is the only category that can lead.
2. **Feature updates** — everything else that moved. A count, not a list.
3. **Exposure changes** — a port newly published, an admin UI reachable, a
   certificate expiring inside 30 days.
4. **Backup freshness** — last successful run, size relative to previous,
   and when a restore was last actually tested.
5. **Capacity** — disks over 80%, and anything growing fast enough to hit
   full before the next sweep.

## Judge before writing

- **Separate security from features.** "14 updates available" is noise.
  "One of these closes a CVE in your reverse proxy, the other 13 can wait"
  is a decision. If you cannot tell which is which, say so rather than
  implying all 14 are urgent.
- **A version gap is not automatically a problem.** Pinned-and-working is a
  legitimate state. Flag it when the gap carries a security fix, breaks a
  supported-version boundary, or blocks something else.
- **Never invent a CVE.** If you know a fix exists, name it. If you only
  know the version moved, say that instead. A fabricated advisory is worse
  than no sweep.
- **Repeat findings are a count, not a paragraph.** If the same 3 updates
  were pending last week, say "same 3 as last week" and move on.
- **If nothing needs them, lead with that** and keep it to two lines. A
  sweep that manufactures work to justify itself gets ignored, and then the
  real one gets ignored too.

## Shape

```
Nothing urgent this week.

Worth doing
- Traefik 3.1.2 -> 3.3.4: one of those releases closed a header-parsing
  CVE. Everything else pending is feature-only.

Backups
- Ran 7/7 days, last night 4.2 GB (in line with the week).
- Last actual restore test: never. Worth 30 minutes to change that.

Also: 13 other updates pending, none security. Disks all under 60%.
```

Name the service and the version. Link the advisory when you have one.

## Never

No connections, no commands, no changes. If the sweep surfaces something
that needs action, describe the action and leave it to the reader.
