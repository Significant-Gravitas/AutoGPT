---
name: emil-design-system-docs
description: "Write design system documentation an AI agent can follow — a DESIGN.md or docs folder covering when to use which component, which variant applies where, and which tokens are allowed, generated from the code that actually exists. Use when creating or updating design system docs, component guidelines, or token documentation, or when an agent keeps hardcoding colors, inventing variants, or picking the wrong component. Triggers on: DESIGN.md, design docs, design system documentation, component guidelines, token documentation, document the design system, document our components, design rules for agents, CLAUDE.md design section, AGENTS.md design rules, agent invented a variant, agent hardcoded a color, agent picked the wrong component."
---

# Design System Docs

Tokens and components define what exists. Docs define when to use what. Write them for a new engineer who joined yesterday and can never ask you questions — that is exactly what an agent is. Every decision the docs leave out gets replaced by a guess, and a guess looks exactly like a decision.

## Process

Five steps, in order. Never skip step 1 — docs written from memory instead of code are how agents learn variants that don't exist.

### 1. Inventory the code first

Read the actual system before writing a single line of documentation:

- **Tokens** — find the token files (globals.css, theme files, Tailwind config). List every semantic token (`--bg-surface`) and the primitive backing it. Note tokens that only exist as raw values — they'll need a rule, not a mention.
- **Components** — list every shared component, its variant and size unions, and the props that change appearance. The union type is the source of truth for what exists.
- **Usage** — open two or three real call sites per component to learn the de facto default variant and the patterns worth blessing or banning.

Every name in the docs must trace to a file. If you can't point at the line a variant comes from, it doesn't go in the docs — a documented variant that doesn't exist teaches every future agent to invent it.

If there is nothing to document — no tokens, no shared components, ad-hoc styles everywhere — stop and say so. Docs can't define a system that isn't there. Offer to set up semantic tokens and base components first, then document them.

### 2. Pick the structure

```
How big is the system?
 ├── One token file and ≤5 components → a single DESIGN.md at the repo root
 └── Bigger → a docs folder
      ├── a root file → the system's character in 2–3 sentences,
      │   plus an index of the granular files
      └── one file per topic → colors.md, typography.md, buttons.md, forms.md
```

Small files are the point. A 2000-line design bible gets skimmed and half-forgotten, by agents and humans alike, and granular files mean an agent only loads what the current task needs. Keep the root file under 100 lines and every topic file under 150; when a file outgrows that, split it by topic instead of trimming the rules.

### 3. Write each component with this exact template

Four parts, always in this order:

````md
## Button

Use for actions. Navigation styled as a button is a Link.
Actions in tight spaces use IconButton.

Variants: `primary`, `secondary`, `danger`. Nothing else exists —
an unlisted variant is a bug, not an option.

- `primary` — the single most important action on a screen. One per view.
- `danger` — destructive or hard-to-undo actions only.
- `secondary` — everything else. This is the default.

```tsx
// Correct
<Button variant="secondary">Cancel</Button>

// Incorrect — ghost doesn't exist in this system
<Button variant="ghost">Cancel</Button>
```
````

1. **When to use it, and when to use the alternative.** Always name the alternative — "use a dropdown for actions, a combobox for dropdowns with search". Without the alternative, the agent has a description; with it, the agent has a choice it can make.
2. **Which variant applies where, and which is the default.** The default matters most: most drift is an agent promoting everything to `primary`.
3. **One correct and one incorrect example.** Real code pulled from the codebase, not invented. Models follow examples far better than prose, and the incorrect example inoculates against the exact mistake agents make most.
4. **A decision flowchart when a choice has three or more options** — a tree gets walked the same way every run, a paragraph gets interpreted:

```md
Is it the single most important action on the screen?
 ├── Yes → variant="primary"
 └── No
      ├── Is it destructive, or hard to undo?
      │    └── Yes → variant="danger"
      └── Default → variant="secondary"
```

For token docs, the load-bearing rule is which layer is allowed: components use semantic tokens (`--bg-surface`), never primitives (`--gray-100`) and never raw values. Then a flowchart per role — "what background do I use", "what text color do I use" — built from the semantic tokens found in step 1.

### 4. Apply the language rules

- **Numbers and absolutes, no hedge words.** "Use small text sparingly" gives the agent nothing to act on — what is sparingly? "Body text is 16px, captions are 14px. Never go below 12px" is a rule it can follow and you can call out when it's broken. Delete every "reasonably", "tasteful", "where appropriate", "try to avoid" — replace each with a number or a never/always.
- **Every non-obvious rule carries its why.** A bare rule gets applied blindly, including where it shouldn't be. A rule with reasoning generalizes to cases the docs never mention: "One `primary` per view — two primary buttons means the screen has no hierarchy."
- **Every line must change what the agent does.** Don't explain what a button or a token is; the agent knows. Filler dilutes the rules that matter, because attention is spread across everything written.

### 5. Verify by prompting

Docs are a hypothesis until an agent has run against them. Ask for a real screen in one sentence — "Build a settings screen for notification preferences. Use our existing components." — then read the drift:

| The drift | The hole | The fix |
| --- | --- | --- |
| Hardcoded a color | Token rule isn't findable or isn't strict | Move the layer rule into the root file |
| Invented a variant | Docs list variants without "nothing else exists" | Close the set explicitly (and tell the user if the union type isn't strict) |
| Picked the wrong component | No "when to use it vs the alternative" line | Add the comparison for that pair |

Fix the docs, never just the output — correcting the output fixes one screen, patching the doc fixes every screen after it. Repeat until a fresh prompt produces a screen with zero drift.
