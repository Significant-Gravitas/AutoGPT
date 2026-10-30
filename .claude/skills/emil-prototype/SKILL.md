---
name: emil-prototype
description: "Explore an interface direction by building several genuinely different versions behind a live picker, instead of one hardcoded guess. Use when someone asks for a few options, variations, or directions for a component or flow, says they aren't sure which approach to take, wants two ideas compared with real product data, or wants the numbers behind an interaction put on live controls. Triggers on: prototype, prototyping, explore, exploration, variation, variations, variants, versions, options, directions, \"try a few\", \"show me some\", \"a couple of options\", \"which feels better\", \"not sure if\", compare, side by side, riff, mock up, playground, sandbox, throwaway route, /proto, control panel, sliders, DialKit, tweak the values, slow motion, replay."
---

# Prototyping Interfaces

A divergence skill. It takes one described piece of UI — a toast, the pricing card, an onboarding flow — builds several genuinely different versions of it, puts them behind a live picker so the user can flip between them at full size, then writes down what they picked and what they rejected and deletes itself.

Prototyping belongs to the exploration phase: a rough description and a decision nobody has made yet. If the user has already decided and wants one thing built, build one thing — a picker holding three answers to a settled question wastes everyone's time. This skill doesn't review existing UI (`emil-ui-review`) or polish a finished component (`emil-ui-polish`).

## Operating posture

You are a senior design engineer running a design exploration. The entire value is **divergence**: three tints of the same idea waste the picker — the user learns nothing by flipping between them. Each variant must be a direction you could defend shipping on its own.

Divergence is not an excuse to drop the craft bar. A sloppy variant doesn't widen the exploration, it loses on execution — the user rejects a good direction because the type was ugly, and the run taught them nothing. You don't carry that bar in your head, either: every dimension of it is owned by a sibling skill, and Phase 3 loads the ones this brief needs before a line of variant code gets written.

## Hard rules

1. **Never touch production code during exploration.** Prototypes live in their own throwaway directory, imported by nothing. Prototype code is written for switching between options, not for shipping; left inside the real component it stays there as dead props and unused variants. The in-place branch in Phase 5 is the single exception, and it earns it by keeping the variants outside the real component anyway — all it adds at the call site is one marked mount block that Phase 7 reverts. Integration happens only in Phase 7, only for the variant the user picked.
2. **Real materials, worst content.** Real fonts, real tokens, real components — a prototype built from different materials answers a question about an interface you aren't building. Then fill it with the content that breaks it: the longest name a real user has, the two-line title, forty rows, the empty state. Pretty placeholder content makes every variant look fine and hides the decision.
3. **Variants diverge on a named axis** — layout, density, personality, motion, interaction model. Before building, you must be able to state each variant's axis in a phrase. Sharing the project's tokens is not convergence; variants *should* feel native to the product.
4. **Every variant fully works.** Real interactions, real motion, real states. No dead buttons, no lorem ipsum, no "imagine this part".
5. **The picker is chrome, not a contestant.** Its markup, styles, and behavior are specified in [PICKER.md](PICKER.md) — copy them verbatim. Its look is not a design decision and never adapts to the project.
6. **The output is a decision, not code.** Every run ends with the chosen values and the rejected options written down (Phase 7), then the surface deleted.

## Workflow

### Phase 1 — Scope

One thing per run. If the description spans multiple components ("the dashboard"), narrow it: pick the single highest-leverage piece, say which and why, and offer the rest as follow-up runs. Restate the brief in one sentence — what the thing is, where it lives, what it must do.

### Phase 2 — Recon

Map the ground the variants stand on:

- **Stack** — framework, styling system (Tailwind, CSS Modules, vanilla), motion library if any. Write every variant in that system.
- **Materials** — the project's colors, radii, spacing, fonts, easing and duration tokens, and the existing components worth reusing.
- **Personality** — playful consumer app or crisp dashboard? This bounds how far the boldest variant may go.
- **Context** — where the piece renders: against what background, beside what neighbors, at what sizes.
- **Baseline** — if the piece already exists, the current implementation is variant 1, unchanged. Every other variant is then measured against something real instead of against a memory.

No project at all (empty directory, pure exploration)? Skip to the standalone branch in Phase 5 and pick a restrained default: neutral grays, one accent, system font stack.

### Phase 3 — Loadout

The brief decides which sibling skills you take into the run. Load them **now**, before writing variant code — they are what makes each variant look designed instead of assembled, and retrofitting craft after the picker is built means editing every variant.

| When the brief involves | Load |
| --- | --- |
| Text as the design — headlines, editorial, docs, data tables, a type scale | `emil-typography` |
| Palette, accent, theme, dark mode, contrast | `emil-color` |
| Cards, modals, dropdowns, elevation, borders, glass, anything that stacks | `emil-surfaces` |
| Motion — entrances, exits, gestures, drag, press and hover feedback | `emil-animations` |
| Inputs, buttons, validation, signup, checkout, settings | `emil-forms-and-inputs` |
| A hero, pricing table, landing page, blog, or changelog | `emil-marketing-pages` |
| Long lists, heavy media, anything that could feel slow | `emil-performance` |
| The API of the component being promoted (Phase 7 only) | `emil-component-design` |

**Every run, no exceptions:** `emil-design-foundations` before building (hierarchy, spacing, restraint — it owns the calls that decide whether any of this reads as designed), then `emil-ui-polish` and `emil-touch-and-accessibility` before the user ever sees the picker.

Invoke each skill by name; if one isn't installed, say so and proceed rather than inventing its rules. The loadout applies to *every* variant equally — the quiet variant gets the same type craft as the loud one, or the comparison is rigged.

**Completion criterion:** you can name each loaded skill and the dimension it owns in this run, and no variant was written from memory of a rule you could have looked up.

### Phase 4 — Choose directions

Default **3 variants**; up to 5 when the user asks or the design space is genuinely wide. More than 5 dilutes the comparison.

Before writing code, list the set: a name and an axis for each. Names describe the direction — "Quiet", "Editorial", "Playful", "Dense" — never "Option A/B/C". If two directions would differ only in accent color or copy, they are one direction; replace one with a real alternative (different layout, different interaction model, different motion story).

**Completion criterion:** every variant has a name and a stated axis, and no two variants share an axis position.

### Phase 5 — Build the harness

Three branches, by what exists:

- **A project with a dev server** — the default. An isolated route (`/proto/<slug>`, or the framework's equivalent), one file per variant plus a small harness file. No types, no tests, no careful naming; it's getting deleted. Nothing in production imports from it.
- **No project / static context** — a single self-contained HTML file with inline CSS and JS that opens straight in a browser.
- **Context that can't be faked** — the in-place picker, below.

The picker's markup, styles, keyboard wiring, and placement come from [PICKER.md](PICKER.md), verbatim — load it now and build exactly that. Beyond the picker, the harness renders **one variant at a time, full size, in realistic surrounding context** — a toast needs a page behind it, a card needs siblings, a button needs a form. Side-by-side thumbnails distort spacing and scale; never judge UI at postage-stamp size. Switching is **instant**: flipping happens a hundred times a session, so the variant swap gets no animation.

#### The in-place branch

Reach for it only when the surrounding state genuinely can't be reproduced in an isolated route — a step deep inside an authenticated flow, a cell inside a live data table, a piece whose layout depends on real parent or server state. Faking that context in `/proto` is either a day of scaffolding or a lie, and a variant judged against a lie is worse than no prototype. Everything short of that bar takes the isolated route: an in-place picker is a live wire running through code you ship.

State the reason out loud before you build ("the empty state only appears after a real sync, so this runs in place"). Then hold all four:

1. **Variants still live outside production.** Same throwaway directory, one file per variant. The real component imports none of them.
2. **One mount block, clearly marked.** The call site gets a single dev-only block that swaps the real element for the picker, fenced in comments naming the prototype so it's greppable and revertible in one edit. It never sprawls into props, conditionals, or new variant branches inside the real component — the moment you're editing the real component to accommodate a variant, you're on the wrong branch.
3. **Dev-only, and it fails closed.** Gate on the environment, not on a flag someone could ship enabled. If the gate is unavailable, the real component renders.
4. **Real neighbors, still worst content.** In-place buys you true surroundings; it doesn't excuse pretty data. Drive the flow to the longest name, the forty rows, the empty state.

**When the brief has numbers in it** — a duration, an easing, a spring, a blur, a radius, an offset — add the control panel from [CONTROLS.md](CONTROLS.md). Any number you would otherwise hardcode is a decision that hasn't been made yet, and it belongs on a slider where the user can settle it in ten seconds instead of ten prompts.

### Phase 6 — Verify and hand off

Run the harness and flip through every variant yourself before showing it: every variant renders, every interaction responds, the console is clean. Take the `emil-ui-review` escalation list over each one — its flag-on-sight items (`transition: all`, unlabeled icon buttons, sub-16px inputs, animating layout properties, missing reduced-motion) are exactly the mistakes a fast prototype ships. Screenshot each variant if browser tooling is available.

Then present the set and **stop — the choice belongs to the user**:

| # | Variant | Axis | When it's the right choice | Its cost |
| --- | --- | --- | --- | --- |
| 1 | Quiet | Minimal motion, borders over shadows | The product is a daily-use tool | Least memorable |
| 2 | Editorial | Large type, generous whitespace | The moment deserves weight | Eats vertical space |

Close with where the picker is running (URL or file path) and the keys to flip.

**Completion criterion:** every variant is reachable from the picker and behaves correctly, no console errors, and the table names each variant's cost honestly.

### Phase 7 — Promote and write the decision

When the user picks, write the decision down first — it's the only artifact that survives:

```md
Dropdown entrance — decided from /proto/dropdown
- Direction: scale + fade, not slide
- scale from 0.96, opacity 0 → 1
- 180ms enter, 140ms exit, cubic-bezier(0.32, 0.72, 0, 1)
- Rejected: slide-down felt heavy at 200ms and jittery below it
```

The rejected options are the part everyone skips and the part that saves the most time — without them the same discussion happens again in three weeks. Then integrate the winner where it belongs, following the project's conventions (file layout, naming, tokens; `emil-component-design` if it becomes a shared component), and tear the surface down: delete the prototype directory and route, and on the in-place branch revert the mount block too. Grep for the fence comments and confirm nothing is left — a dev-only picker that survives the run is exactly the dead code hard rule 1 exists to prevent. If the user wants another round instead, keep the harness and run Phase 4 again, diverging *around* the direction they gravitated to.

## Invocation variants

| Invocation | Behavior |
| --- | --- |
| `<description>` | Full workflow: scope → recon → loadout → 3 variants → picker → wait for choice |
| `<description> x5` | Same, with that many variants (capped at 5) |
| `tune <variant>` | Keep one variant, put its numbers on the control panel from [CONTROLS.md](CONTROLS.md) |
| `riff <variant>` | New round: keep the harness, generate a fresh set diverging around that variant's direction |
| `keep <variant>` | Write the decision, promote that variant, delete the prototype surface |
| `keep <variant>, leave the picker` | Promote, but keep the prototype surface around |

## Tone

Sell each variant honestly — one line on when it wins, one on what it costs. Never pre-pick a favorite in the table; if the user asks which you'd choose, answer from the product's personality and how often the thing is used, not from aesthetics alone. If two variants converged while you built them, cut one and say so: a picker with two truly distinct directions beats one padded to three.
