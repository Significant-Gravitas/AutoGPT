---
name: "product-ai-feature-scoping-and-evals"
description: "Use when the user names an AI feature, an agent, a prompt, retrieval, evals, or model quality work: the scoped spec with a prompt set, a four-rung quality ladder, a scoreboard read over three runs, and the safety gates."
triggers: ["ship an AI feature", "eval plan", "build the prompt set", "model quality bar", "scope an agent feature", "red-team prompts", "is the model good enough", "grade the model outputs"]
version: "1"
---

# AI feature scoping and evals

Use this when the user names an AI feature, an agent, a prompt, retrieval, evals,
or model quality work. The output is a scoped spec whose quality claim is
measured rather than asserted.

## What you need first

The job the AI does, the users, the quality bar, and the cost of being wrong.
No quality bar means you propose one from the failure cost and mark it
INFERENCE. No eval data means you build the prompt set before any talk about
which model to use.

## Scope it, then measure it

1. Say back the job in one line: the input, the output, and who is harmed when
   it is wrong. High-harm outputs get human review by default — say so early.
2. Build the prompt set first: twelve to twenty prompts across the real kinds
   of input the feature will see — happy path, edge, adversarial, and
   out-of-scope — and mark the top five that must never regress.
3. Set the quality bar as a ladder with four rungs: nailed it, usable with
   caveats, wrong but harmless, harmful or fabricated. Every rung needs an
   example, not an adjective.
4. Capture before you judge: collect the model's exact outputs first, then
   grade each against the ladder with exact quotes. Never grade from memory.
5. Score it on a board: counts by rung, broken down by prompt kind, with a
   three-run trend rule. One run is noise; three runs is a read.
6. Spec the trade-offs: quality against latency, cost per task, and safety.
   Name the red-team prompts and the audit trail the feature keeps. For
   retrieval-augmented generation, spec what gets retrieved, from where, and
   what happens when nothing relevant comes back.

## Output

The dated eval plan with the prompt set, the ladder, and the board, plus the
feature spec it feeds. Offer to rerun on a yes whenever the model or the
prompt changes. Point at an existing open-source eval harness for the running
of it rather than building one.

## When the outputs are not measured

Say the number is not measured and refuse to quote one. Never claim a quality
number you did not collect, and never ship a high-harm output without a gate
and the owner's yes.
