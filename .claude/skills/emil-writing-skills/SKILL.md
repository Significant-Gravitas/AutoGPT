---
name: emil-writing-skills
description: "How to write skill files that actually change what an agent does — encode process not output, pair every rule with its why, be strict instead of vague, cut lines that don't change behavior, keep skills focused, and test by running. Use when creating a new SKILL.md, reviewing or refining an existing skill, packaging taste or conventions into a reusable skill file, or diagnosing why a skill isn't affecting the agent's output. Triggers on: skill, SKILL.md, skill file, write a skill, create a skill, improve a skill, agent instructions, CLAUDE.md, AGENTS.md, system prompt rules, taste, conventions, decision tree, \"the agent ignores my skill\", \"skill doesn't work\", package my preferences, transfer taste."
---

# Writing Skills

Agents are probabilistic — ask the same question twice and you get two different answers. A skill fights that not by forcing the same *output* every time, but the same *process*. The outputs should still differ, every context is different, but the reasoning becomes yours and stays the same every run. Write a skill like you're guiding a less experienced designer who happens to be very fast.

## Core Principles

### 1. Encode process, not output

Give the agent a decision procedure it can walk every single time, not a description of the result you want. Decision trees are ideal:

```md
## Easing Decision Flowchart

Is the element entering or exiting the viewport?
├── Yes → ease-out
└── No
    ├── Is it moving/morphing on screen?
    │   └── Yes → ease-in-out
    └── Is it a hover change?
        ├── Yes → ease
        └── Is it constant motion?
            ├── Yes → linear
            └── Default → ease-out
```

Without this, the agent picks whatever feels right that day. With it, every run walks the same tree.

### 2. Write down the why

When a rule doesn't cover a situation, the agent guesses — and its guess looks exactly like a decision. A rule without reasoning gets applied blindly, including where it shouldn't be. A rule *with* reasoning generalizes, because the agent can extend the logic to cases you never wrote down.

```md
<!-- Bad — rule only -->
Start scale animations from 0.95.

<!-- Good — rule + reasoning -->
Start scale animations from 0.95, not 0. Elements appearing from
nothing feel unnatural — real objects always have a visible shape.
The higher the initial value, the more gentle and elegant the entrance.
```

### 3. Be strict

Vague language is the most common failure mode. Words like "reasonably", "tasteful", and "where appropriate" give the agent nothing to act on, so it falls back to its defaults — those sentences might as well not be there.

```md
<!-- Bad — unactionable -->
Try to keep animations reasonably short. Use tasteful easing curves
where appropriate.

<!-- Good — concrete and strict -->
UI animations stay under 300ms. Exit animations are ~20% faster than
entrances. Larger elements animate slower than smaller ones.
```

Word choice carries weight: "never", "always", and "strict" anchor behavior; "try to avoid" does not. The strictness can feel like it leaves no room for creativity, but the creative part is still yours — the skill just stops the agent from guessing at the parts you've already figured out.

### 4. Every line must earn its place

Go through the skill sentence by sentence and ask one question: does this line change what the agent does? The agent already knows what CSS transforms are. It knows what a modal is. Explaining these things doesn't just waste space — it dilutes the lines that matter, because the agent's attention is spread across everything you wrote. A great skill, like a great animation, is defined as much by what you leave out.

### 5. Keep skills focused

One skill per aspect of the interface: one for animation, one for typography, one for layout. Small, focused skills beat one giant one.

### 6. Test it by running it

Test a skill like you'd test an animation — by running it. Unsure whether a line matters? Run the skill with the line and without it, and compare the output. This is the same loop that builds taste: create, notice what feels off, articulate why, refine. Except now the thing you're refining is the document that carries your taste, and every improvement compounds across everything your agents build from then on.

## Common Mistakes

- **Describing outcomes instead of process.** "Animations should feel smooth" forces the agent to guess. A decision tree or concrete constraint doesn't.
- **Rules without reasoning.** Bare rules get applied blindly in places they don't belong, and can't generalize to cases you didn't write down.
- **Hedge words.** "Reasonably", "tasteful", "where appropriate", "try to avoid" — none of these change behavior. Replace with numbers and "always"/"never".
- **Explaining what the agent already knows.** Definitions of CSS properties or UI patterns dilute the lines that actually encode your taste.
- **One giant skill.** A do-everything skill spreads attention thin; split it by aspect.
- **Shipping untested.** A skill is a hypothesis until you've run it and looked at the output.

## Checklist

- [ ] Rules are decision procedures the agent can walk the same way every run
- [ ] Every rule that isn't self-evident carries its why
- [ ] No hedge words — constraints are numeric or absolute ("always", "never")
- [ ] Every line changes what the agent does; everything else is deleted
- [ ] Skill covers one aspect, not everything
- [ ] Tested by running with and without the lines in question
