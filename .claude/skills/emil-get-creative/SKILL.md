---
name: emil-get-creative
description: "Get unstuck on creative visual work. When the user is stuck, fishing for inspiration, or holding only a vibe — mood words, a feeling, \"something like…\" — instead of a spec, don't prompt for finished outputs. Walk the pipeline instead: research the visual language behind the vibe until it has names, compress the findings into rules, then build a parameterized generator — a canvas driven by a seeded config, a panel of knobs, click-to-regenerate, and a save button that writes the winning configuration to JSON. Use when the user invokes /get-creative, says \"I'm stuck\", \"I need inspiration\", \"I can't picture it yet\", describes a mood rather than an image, wants endless variations of one visual idea, or has rejected a round of prompted variants."
disable-model-invocation: true
---

# From Vibe to Tool

Creative work has to produce something new, and prompting alone is bad at new — the model falls back on known patterns, the user rejects the result, asks for "a few more options", and the loop burns time and tokens without converging. The escape has two halves. **Research** turns a vibe the user can't articulate into rules they can point at — most looks people reach for were shaped by a real process, and the constraints of that process *are* the style. **A generator** then plays those rules: every variation one click, every taste decision a knob, the keeper reproducible forever. You build the tool; the tool builds the thing.

Two neighbours to keep straight:

- `emil-prototype` compares 3–5 discrete, hand-built directions for product UI behind a picker. If the question is "which of these approaches?", run `emil-prototype`.
- `emil-build-a-tool` builds a tool aimed at a **target the user can already point at** — a reference, a screenshot, a Figma frame — and dials it in until it matches. If the user knows exactly what the output should look like, run `emil-build-a-tool`.

This skill is for the user who *doesn't* know yet — they'll recognize the right thing only when they see it.

## Phase 0 — Route

Walk this before doing anything:

Can the user point at or precisely describe the exact output they want?
├── Yes → build that output directly; if it's too intricate to land by
│         prompting, it's a matching problem → `emil-build-a-tool` skill
└── No
    ├── Is it product UI with a few nameable directions? → `emil-prototype` skill
    ├── Can they already state the idea as rules — a construction they could
    │   explain to a stranger? → skip research, go to Phase 2
    └── Do they only have a vibe — mood words, an era, a feeling, "I'm
        stuck", "I want something…"? → Phase 1. Never interrogate them into
        a spec first; the research exists precisely because they can't
        answer those questions yet.

**The two-round rule.** If you are mid-conversation and about to produce a *second* round of prompted variants of the same visual idea, stop and propose this pipeline instead. Two rejected rounds is the signal that the user is searching a space, not requesting an item — every further prompted round costs more than the tool would.

## Phase 1 — Research the visual language

The goal is to earn the vocabulary: leave this phase with words for what the user could only feel on the way in.

1. Capture the vibe in the user's own words, verbatim. Don't translate or upgrade their language yet — "warm", "old-timey", "kind of hand-done" are data, and rewriting them into your nearest cliché is how the result drifts off-vibe.
2. Research the tradition behind the vibe, and always answer three questions:
   - **What process made this look?** Printing method, material, tool, medium, era of technology. The look's signature quirks are almost always constraints of that process, so reproducing the constraints reproduces the style.
   - **What did the process do to color, texture, and composition?** Limited palettes, characteristic imperfections, how elements interact where they meet.
   - **What are the recurring patterns and motifs called?** Names are handles — a named pattern can be looked up, requested precisely, and bound to a knob. An unnamed one can only be gestured at.
3. When the vibe spans more than one thread, run the threads as parallel research agents rather than one long serial pass. Spend tokens here without guilt — research is the part of this workflow worth paying for, because the paragraph it produces is the foundation everything after is built on. A cheap research pass produces generic rules, and generic rules produce the exact prompted-variant mush the user came here to escape.
4. Compress the findings into 5–10 rules stated as constraints ("no more than N colors", "X darkens where it crosses Y", "edges are never clean"), plus a list of named patterns. Show this to the user and let them strike or keep lines **before any code exists**. This is the taste checkpoint: rules are cheap to edit, renders are not.

## Hard rules for the generator

1. **Output is a pure function of the config.** One `render(config)` where config is a plain object containing every parameter *and the seed*. All randomness comes from a seeded PRNG (e.g. mulberry32) initialized from `config.seed` — never a bare `Math.random()` in the render path. Why: the save button is the whole point of the tool, and it's a lie if the saved config can't reproduce the exact image the user saved.
2. **A new variation is one click, not one prompt.** Clicking the canvas assigns a fresh random seed and re-renders. The economics of the workflow depend on generation being free — if seeing another variation requires talking to you, the tool has failed.
3. **Every taste decision is a knob.** Any number you'd otherwise hardcode — density, amplitude, palette index, line weight, jitter — is a decision the user hasn't made yet, and it goes in the panel. Fixed values are only for the rules from Phase 1; those are what make the idea itself. Give every knob a real min/max you've tested; a slider that can reach a broken render teaches the user to distrust the tool.
4. **Save writes the full config.** One button that persists the *entire* current config — seed included — as pretty-printed JSON. Copy-to-clipboard at minimum; when there's a dev server, also a dev-only endpoint that writes it to a JSON file in the project (the file the real artifact will read). Show the written path in the UI after saving. Why a file and not just clipboard: the production artifact imports the file, so saving *is* shipping the decision.
5. **The tool is scaffolding; the config and render function are the deliverable.** The generator lives on an isolated route (`/lab/<slug>` or the project's equivalent) and nothing in production imports it. What gets promoted is the saved JSON plus the render function — or a static export (SVG/PNG) when the artifact doesn't need to stay live.

## Phase 2 — Sort the rules into buckets

With the researched (or user-stated) rules in front of you, sort every line into three buckets and state the result in a few lines before building:

- **Rules** — what makes it *this* idea and never changes. Hardcoded.
- **Modes** — discrete sub-directions worth switching between (which motif carries the piece; which corner of the tradition). Tabs or a select, each mode allowed its own knobs.
- **Knobs** — every continuous value (scale, count, palette, weight, jitter, drift). Panel controls, each with tested bounds and a good default.

If the knob bucket is empty, the idea is fully determined and you didn't need a generator — go build the output.

## Phase 3 — Build the tool

One sitting, no ceremony — it's a lab, not a product:

- Isolated route, one canvas rendered large. The output is judged at real size; thumbnails hide the texture that makes or breaks generative work.
- Mode tabs, the knob panel, click-to-regenerate on the canvas, and the seed displayed somewhere visible (a seed you can read is a variation you can talk about).
- If you use a panel library like leva: keep every row unconditional. Conditionally rendering rows inside folders breaks the panel layout — show all knobs and let irrelevant ones no-op per mode, or split folders per mode.
- **Save** (writes config JSON + clipboard) and a small rail of saved configs that re-applies one on click. Taste is comparative — the user decides between the current variation and the one from four clicks ago, so recalling a saved state must be instant.
- Everything re-renders live as knobs move. A generator you have to "apply" changes to loses the tight see-tweak-see loop that makes it work.

Then hand it off in two lines: the URL, and the verbs — *click the canvas for a new one, knobs to tweak, save to keep.* Then stop. The exploring is the user's; hovering with suggestions defeats the purpose of having built them a tool.

## Phase 4 — Harvest

When the user says a saved one is the keeper:

1. Read the saved JSON — never eyeball-copy values from a screenshot of the panel.
2. Wire it into the real artifact: production imports the config file and the (cleaned-up) render function, or you export the render as a static SVG/PNG if it doesn't need to be live.
3. Keep the lab route unless the user asks to delete it — a generator gets reused the next time the same visual language needs another asset. If it stays, note the route and its saved-config location wherever the project tracks such things.

## Invocation variants

| Invocation | Behavior |
| --- | --- |
| `<vibe or idea>` | Full pipeline: route → research → rules → tool → hand off |
| `research <vibe>` | Research only: return the rules and named patterns, build nothing |
| `harvest` | Read the latest saved config and wire it into the real artifact |
| `harvest <path/to/config.json>` | Same, from a specific saved config |
| `more knobs` | Promote hardcoded values in the current tool to panel controls |
| `new mode <description>` | Add a discrete sub-direction to the existing tool |
