---
name: "roadmap-prioritisation"
description: "Rank candidate work with a stated method, shown inputs, and marked guesses."
triggers: ["prioritise roadmap", "RICE score", "what to build next", "rank features", "roadmap planning"]
version: "1"
---

# Roadmap prioritisation

Use this when the team must choose what to build next from a list of options.

## Agree the method

Use the team's chosen method, such as RICE, impact against effort, or goals
first. If none is set, suggest one and wait for agreement. Write down what
each input means and where it will come from, for example reach from usage
data and effort from engineering.

## Score with sources

For each item show every input, its source, and its date. Mark inputs that
are guesses. Effort comes from the people who will do the work; if they have
not given one, leave it blank rather than filling it in. Note links between
items and anything already promised to a customer.

## Ranked list and caveats

Return:

1. the ranked list with scores and the one-line reason for each;
2. items whose rank would change if a guess were wrong;
3. what the method leaves out, such as strategy or tech debt;
4. the decision the team needs to make, and by when.

Do not move items on the roadmap or tracker yourself; you recommend and the
team decides. Never invent a score input, present a guess as an estimate, or
promise a date.
