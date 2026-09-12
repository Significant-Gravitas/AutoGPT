---
name: daily-repo-brief
description: Produce a short read-only morning brief on a repository - what merged, what broke, what is stale, and what is waiting on the reader specifically. Use for a daily or weekly repo digest, a catch-up after time away, or when asked what changed and what needs attention.
triggers:
  - daily brief
  - what changed
  - catch me up on the repo
  - what needs my attention
---

# Daily repository brief

One message, read in under thirty seconds, that answers: has anything broken,
and is anything waiting on me. Everything else is optional detail.

This skill is **read-only**. It never merges, closes, comments, labels, or
pushes. That is what makes it safe to run on a schedule.

## This usually runs unattended

The brief is the expert's scheduled activity, so most runs have nobody
watching. **Never make a question the whole output.** If something is
missing, produce the brief you can and put the gap in it:

- **No repo identified.** Say so as the brief itself and name what you need
  once: "No repository is connected, so I can't produce a brief. Connect one
  or tell me the `org/repo` and I'll run from tomorrow." Then stop. Asking
  "which repository?" and nothing else means a scheduled run delivers
  nothing at all, every morning, silently.
- **State supplied in the message.** If the reader has given you the numbers
  directly, render the brief from those and label it as reported rather than
  verified. Do not refuse to write it because you could not confirm the
  figures yourself, and do not silently present them as checked.
- **Repo reachable but a source failed.** Brief on what you have and list
  what you could not read, so a partial answer is never mistaken for a
  complete one.

## Gather

Scope to the window since the last brief (default: previous 24h, Monday
covers the weekend).

1. **Merged** — PRs merged in the window, with author and one-line effect.
2. **Broken** — failing CI on the default branch, and any revert. This is the
   only category that can lead the brief.
3. **Waiting on the reader** — PRs where they are a requested reviewer, or
   their own PRs with unresolved threads or failing checks.
4. **Stale** — open PRs with no activity for 7+ days, and unassigned issues
   older than 14 days. Count them; list only the ones that block something.
5. **New** — issues opened in the window that look actionable.

## Judge before writing

The value is in what you leave out. Apply these:

- **A merged PR that changes nothing for the reader is a line, not a
  paragraph.** Group them: "4 merged (docs, deps, two block fixes)."
- **Never list something as needing attention without saying why now.** "9
  days waiting and blocking two other PRs" is a reason; "still open" is not.
- **A stale PR that nothing depends on is a count, not an entry.**
- **If nothing needs the reader, say that in the first line.** A brief that
  manufactures urgency to justify itself trains people to skip it.

## Shape

Lead with breakage if there is any, otherwise with what needs them.

```
Nothing broken. 2 things need you.

Needs you
- #14287 — waiting 9 days on your review; blocks #14290 and #14301.
- #14312 — your PR, kcze left an unresolved thread Tuesday, no reply yet.

Merged (5)
- #14301 fixes the duplicate-send on retry (Sam)
- 4 others: docs, two dep bumps, a block rename

Stale: 11 PRs untouched 7+ days; none blocking. 3 unassigned issues 14+ days.
```

Link every reference. Name people by their handle where a decision is theirs.

## When the repo is quiet

Say so in one line and stop. "Nothing merged, nothing broken, nothing waiting
on you." Do not pad with the stale count on a quiet day — it is the same
number as yesterday and reads as noise.

## Never

No writes of any kind. If the brief surfaces something that needs an action on
GitHub, describe the action and let the reader take it.
