---
name: "product-prd-and-acceptance-criteria"
description: "Use when the user names a feature to spec, a build decision, user stories, or acceptance criteria: the PRD with a scope line, stories a tester can check, UX direction at wireframe level, the success metric, and the engineering handoff."
triggers: ["write a PRD", "spec this feature", "user stories", "acceptance criteria", "draw the scope line", "what is out of scope", "engineering handoff", "smallest shippable version"]
version: "1"
---

# PRD and acceptance criteria

Use this when the user names a feature to spec, a build decision, user stories,
or acceptance criteria. The output is a PRD engineering can build from without
coming back with questions.

## What you need first

The problem, the users, the goal, and what is explicitly out of scope. No goal
means you write the smallest shippable bet and mark the goal UNKNOWN. No scope
boundary means you draft the outs yourself and get a yes on them before you
write a single story.

## Write it

1. Open with the one-paragraph bet: the user, the job, the outcome, and the
   smallest thing that could prove it. If the MVP needs more than one
   paragraph, it is not an MVP. Put the header table above it — owner,
   participants, status, target release.
2. Draw the scope line: in, out, and later. Every out gets one line on why.
   The outs are as much the spec as the ins.
3. List the assumptions — the technical, business, and user-behavior beliefs
   the spec rests on — each with how it gets validated and by when. Revisit
   them through the project: a belief that fails rewrites the spec.
4. Write the requirements as user stories, one ticket per story: as a
   [persona], I want [action] so that [outcome], plus acceptance criteria a
   tester can check without asking you a question, plus a priority.
5. Note the UX direction at wireframe level: the screens, the happy path, and
   the empty, loading, and error states. Flows over mockups — design owns the
   pixels, and you describe them in words or point at the file in their design
   tool.
6. Name the success metric as baseline to target by date, the instrumented
   events behind it, and the kill line. A PRD with no metric is a wish.
7. List the open questions with owners and dates. No ownerless question ships.

## Output

The dated PRD: header table, the bet, the scope line, assumptions, stories with
acceptance criteria, UX direction, the metric, and open questions. Take one
round of edits, then file the stories as tickets only on a yes. Engineering
gets the handoff with the scope line up top, and never before the owner has
said yes to that scope.

## When the inputs are missing

Write the gap as an open question with an owner and a date rather than filling
it in. Never write a requirement you cannot trace to a user, a bet, or a prior
yes, and never invent API shapes or data the team did not confirm.
