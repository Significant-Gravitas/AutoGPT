---
name: emil-build-a-tool
description: "Analyze what the user is actually trying to build and suggest the tool that gets them there — a means to an end, not the end itself. When prompting for an artifact directly keeps landing on \"closer, but not quite\", stop iterating on the output and build the small tool that produces it: diagnose why the direct path fails, propose the tool that closes that exact gap, put the taste decisions on live controls, and export exactly what the project needs. Use when the user says \"build me a tool for this\", describes a result the agent keeps missing, or has burned a round or two of re-prompting on the same artifact."
disable-model-invocation: true
---

# Build a Tool

The user wants an artifact. You are not going to build the artifact — you're going to build the tool that builds it. Prompting for an intricate result directly is a slot machine: every attempt costs a full round trip, lands "close", and gives the user no way to steer except rewriting the prompt. A tool converts that loop into knobs — the process gets built once, and the taste decisions become adjustments the user makes by eye in seconds.

This skill has two halves, in strict order: **diagnose, then build.** The most common failure is skipping the diagnosis and building a generic playground. A tool earns its existence by targeting the exact reason the direct path fails, so finding that reason is the actual work.

## Phase 1 — Diagnose the gap

Answer these four questions from the conversation before proposing anything. Ask the user only for what you can't infer:

1. **What actually ships?** Name the artifact and where it lives in the project. The tool is never it.
2. **Why can't you build it directly?** The core of the analysis — classify the gap below.
3. **What knows the process?** An app that produced the reference, a named technique, a spec, an existing implementation. The tool should reproduce a process that's known to work, not invent one.
4. **What format does the project need?** The answer comes from what the code has to do with the output — performance, interactivity, file size — never from what's easiest to export.

### Classify the gap

Why does prompting keep missing?
├── The result depends on judgment calls only the user's eye can make
│   (density, balance, color, "when it feels right")
│   → **dial-in tool** — rebuild the process, bind every taste value to a live control
├── The result must match something that already exists
│   (a design, another app's output, a reference image)
│   → **matching tool** — reproduce the process that made the reference,
│     render output next to it at the same size
├── The result exists but in the wrong form
│   (right pixels, wrong format; right data, wrong structure)
│   → **converter** — ingest the source, export exactly what the project needs
├── The result can only be judged in a state that's slow to reach
│   (a specific screen, dataset, viewport, moment in an interaction)
│   → **stage** — a route that puts the work in that state instantly, every time
└── There is no target yet — the user is searching and will know it when they see it
    → exploration, not matching: hand off to the `emil-get-creative` skill

Most real tools combine two branches — a matching tool usually needs dial-in controls, a stage often wants a converter's export. Name the combination you're proposing.

### When NOT to build a tool

- You'd land the artifact in one or two direct attempts → just build it; the tool is overhead.
- The result is fully determined, with no judgment calls left → build it directly; a tool with nothing to decide is a detour.
- Unclear which side you're on → make one honest direct attempt first. One attempt is cheaper than the wrong call in either direction.

**The stop-the-loop rule.** If a round of "closer, but not quite" has already happened, the diagnosis is done for you — that loop *is* the evidence. Say so explicitly and propose the tool, instead of quietly retrying the same prompt in different words. Every further attempt is another paid spin at a target the user can already see.

## Phase 2 — Suggest the tool

Before writing code, pitch it in this exact shape, in a few lines:

- **The gap:** why the direct path fails, in one sentence.
- **The tool:** what it renders, and which branch (or combination) from the taxonomy it is.
- **The controls:** which decisions belong to the user's eye.
- **The export:** the exact format the project will import, and how saving works.

The tool encodes your understanding of the problem, and this is the cheapest moment for the user to correct that understanding — a wrong tool built fast is still wrong. If the user is present, give them a beat to redirect; if they've already told you to go, state the pitch and build immediately.

## Phase 3 — Build it

One sitting, no ceremony — it's scaffolding, not a product:

1. **Isolated route.** `/lab/<slug>` or the project's equivalent. Nothing in production imports from it; what gets promoted later is the saved output plus the cleaned-up render logic.
2. **Rebuild the process, not the output.** Name the technique before writing the render function. If an existing app made the reference, its settings panel is your parameter list — take it as validated by someone who already solved this. A tool that traces the output matches once and can't be adjusted, which defeats the point of building one.
3. **The target stays on screen.** If there's a reference, render the output beside it at the size it will ship at. Matching from memory across a tab switch is how "close enough" ships.
4. **Every judgment call is a live control.** If you'd otherwise hardcode a number you're not sure about, you'd be guessing at the user's eye — bind it instead. Fixed values are only for the rules of the process. Give every control tested bounds; a slider that reaches a broken render teaches the user to distrust the tool.
5. **Instant feedback.** Everything re-renders as controls move. A tool with an "apply" step loses the see-tweak-see loop that justifies its existence. If you use a panel library like leva, keep every row unconditional — conditionally rendered rows inside folders break the panel layout.
6. **Save writes a file.** One button persists the full current configuration as JSON in the project — plus the exported artifact data if that's separate — and shows the written path in the UI. Production imports that file, so saving *is* shipping the decision; values eyeballed off a panel are how the shipped version drifts from the approved one.

Then hand it off in two lines — the URL and the verbs — and stop. The dialing-in belongs to the user; hovering with suggestions defeats the purpose of having built them a tool.

## Phase 4 — Harvest

When the user picks a keeper:

1. Read the saved config and export — never copy values by eye from the panel.
2. Wire the export and the cleaned-up render logic into the real artifact.
3. Leave the tool in place unless asked to remove it. The next asset in the same visual language goes through the same tool, and a tool that's still there is a tool that gets reused.

## Invocation variants

| Invocation | Behavior |
| --- | --- |
| `<the thing you're stuck on>` | Full workflow: diagnose → suggest → build → hand off |
| `diagnose` | Phases 1–2 only: analyze the situation and suggest the tool, build nothing |
| `harvest` | Read the latest saved config and wire it into the real artifact |
| `harvest <path/to/config.json>` | Same, from a specific saved config |
| `more knobs` | Promote hardcoded values in the current tool to controls |
| `export as <format>` | Add or replace the exporter |
