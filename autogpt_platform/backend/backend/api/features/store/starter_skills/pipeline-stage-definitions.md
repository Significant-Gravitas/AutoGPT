---
name: "pipeline-stage-definitions"
description: "Define each pipeline stage with an entry rule, an exit rule, and required proof."
triggers: ["pipeline stages", "stage definitions", "sales stages", "deal stages", "exit criteria"]
version: "1"
---

# Pipeline stage definitions

Use this when reps read stages differently or the forecast leans on stage.

## Review how stages are used

From a dated deal export, count deals by stage, median days in stage, and
conversion to the next stage. Note stages that are skipped, stages where deals
sit for months, and stages that mean different things to different reps. Ask
two or three reps how they decide to move a deal.

## Write the definitions

For each stage write:

- what the stage means, in one line;
- the entry rule: what must be true to move a deal in;
- the exit rule: what must be true to move it on;
- the proof: the field, note, or document that shows it;
- the expected time in stage, from the data, not a target.

Keep the language plain enough that a new rep applies it the same way. Mark
any rule that needs a new field.

## Deals that do not fit

List open deals that would sit in a different stage under the new rules, with
owner and amount. Do not move them; give the list to the sales lead.

Do not change stages, fields, or deals in the CRM yourself. Never set a
probability or a stage from a guess; the sales lead approves the definitions.
