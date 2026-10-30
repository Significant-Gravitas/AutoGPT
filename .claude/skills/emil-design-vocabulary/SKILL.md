---
name: emil-design-vocabulary
description: The precise design and UI vocabulary used on index.how/to/articulate — covering typography, color, iconography, layout, interaction, motion, accessibility, information architecture, copywriting, tools, analysis, and components. Use when reaching for the exact word for a design concept ("what's the term for the space between two specific letters?"), when a UI idea is described loosely and needs its proper name, when choosing between confusable near-synonyms (badge vs tag, tooltip vs popover, opacity vs visibility, kerning vs tracking), or when writing or reviewing copy, specs, or commits and you want exact terminology instead of vague language.
---

# Design Vocabulary

A reverse dictionary for design and UI work. The goal is to go from a fuzzy
description of a concept to the precise, agreed-upon word — and to know the
difference between words that look interchangeable but aren't.

These are the terms from `index.how/to/articulate`: "the words designers use
when they know what they're looking at."

## Source of truth

The authoritative list of terms and definitions is bundled next to this file:

```
vocabulary.md
```

Always pull the exact definition from there — it is the canonical content.
`terms-index.md` (also next to this file) is a flat list of every term name
grouped by category, for fast scanning when you don't yet know which word you're
looking for. If the index and `vocabulary.md` ever disagree, `vocabulary.md`
wins.

## How to use it

**Reverse lookup (the main job — "what's the word for…?")**

1. Read the user's loose description of the concept.
2. Scan `terms-index.md` to find candidate terms, narrowing by category (the
   description usually points at one: spacing between letters → Typography,
   empty area on a page → Layout, a label attached to an element → Components).
3. Open `vocabulary.md` and read the matching definition(s).
4. Answer with: **the term**, its definition, and — critically — any contrasting
   term the definition names, so the user learns the boundary, not just the word.

**Forward lookup ("what does X mean / am I using it right?")**

Find the term in `vocabulary.md`, give the definition, and surface the nearest
confusable term so the user can confirm they mean the right one.

**Disambiguation ("is this a badge or a tag?")**

Pull both definitions and state the distinguishing rule plainly. Many
definitions in `vocabulary.md` are written specifically to draw these lines —
quote that line.

**Encouraging precise usage**

When reviewing copy, specs, PR descriptions, or commit messages, swap vague
phrasing for the exact term where one exists ("the spacing between the letters"
→ "tracking"; "make the empty space bigger" → "more negative space"). Don't
force jargon where plain language is clearer — precision is the goal, not
density.

## Near-synonyms worth disambiguating

These pairs are the ones people reach for the wrong half of. When a request
touches one of them, name the distinction proactively:

- **Kerning vs Tracking** — kerning is between two *specific* characters;
  tracking is uniform across a run of text.
- **Badge vs Tag** — a badge is *attached* and informational; a tag is
  *standalone*, selectable, or removable.
- **Tooltip vs Popover** — a tooltip can't hold interactive content; a popover
  can.
- **Opacity vs Visibility** — opacity zero still occupies space and receives
  pointer events; visibility hidden keeps the space but removes interaction.
- **Ease-in vs Ease-out** — ease-out for things *entering* the screen; ease-in
  for things *leaving*.
- **Modal vs Sheet vs Drawer** — overlay that interrupts vs panel sliding from a
  side edge vs panel pulling up from the bottom.
- **Voice vs Tone** — voice is the constant personality; tone shifts with the
  moment.
- **Chroma vs Saturation** — chroma is the OKLCH, perceptually-accurate
  equivalent of saturation.
- **WCAG vs APCA** — WCAG is the established contrast standard; APCA is the newer
  model that accounts for font size and weight.
- **Variables vs Tokens** — variables are named values; tokens are design
  decisions both design tools and code can reference.

## Categories at a glance

Use these to jump straight to the right slice of `terms-index.md`:

| Category | Covers |
|---|---|
| Typography | How text is set, spaced, and read |
| Color | How color is defined, mixed, and applied |
| Iconography | How icons are drawn and balanced |
| Layout | How space and structure organise a page |
| Interaction | How elements respond to input |
| Motion | How things move and transition |
| Accessibility | How interfaces work for everyone |
| Information Architecture | How content is structured and found |
| Copywriting | How words guide and reassure |
| Tools | How design is made and shared |
| Analysis | How design decisions are measured |
| Components | The building blocks of an interface |

## Examples

**Reverse lookup**

> User: "What do you call the little gap between two specific letters that you nudge by hand?"

Read Typography in `vocabulary.md`. Answer:

> That's **kerning** — the space between two specific characters, adjusted
> manually. Don't confuse it with **tracking**, which is letter-spacing applied
> uniformly across a whole word or block.

**Disambiguation**

> User: "Should this little 'New' label be a badge or a tag?"

> If it's attached to another element and just informational, it's a **badge**.
> But the moment it carries a word like "New" or a status, it starts behaving
> like a **tag** — standalone, and sometimes selectable or removable — and needs
> that treatment instead. The two aren't the same thing.

**Precise usage in review**

> Spec says: "add some space so the lines aren't too cramped."

> The term is **leading** — the vertical space between lines. Worth naming it
> explicitly so it's not confused with **tracking** (horizontal letter-spacing)
> or **gap** (space between flex/grid children).

## Guidelines

- Quote definitions from `vocabulary.md` rather than paraphrasing — the wording
  is deliberate.
- Always surface the contrasting term when one exists. The value of this
  vocabulary is in the boundaries between words, not the words alone.
- Match the user's locale and spelling in `vocabulary.md` (it uses British
  spellings like "centre").
- Don't invent terms. If no listed term fits, say so and describe the concept
  plainly rather than coining jargon.
