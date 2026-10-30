# Typefaces and formats

Which family to reach for, which file to serve, and why two fonts set at identical sizes rarely look identical.

## File formats

Serve `woff2` on the web, full stop. It wraps the font in Brotli compression and enjoys support in every current browser. The older `woff` format compresses less efficiently and matters only if a project genuinely must reach very old browsers. `ttf` and `otf` are the raw desktop containers — no web-oriented compression, noticeably heavier over the wire — so finding one in a `public/fonts` directory is a bug to fix, not a preference to respect.

## The five broad families

Category communicates before any individual font does:

- **Sans-serif** — uniform, unadorned strokes that hold up at small sizes. The default for product interfaces; think Helvetica, Roboto, or Public Sans.
- **Serif** — small finishing strokes give each line a rail for the eye to run along. At home in essays, editorial pages, anything read at length.
- **Monospace** — one fixed advance width for every glyph, so vertical columns align. Code, diffs, tabular IDs.
- **Display** — drawn to be seen big: posters, heroes, mastheads. Often too spindly or too quirky for paragraph duty.
- **Script** — imitates handwriting. A garnish for rare decorative moments, never a workhorse.

CSS's generic keywords cover these loosely; `cursive` maps to script faces and `fantasy` to decorative ones.

## Optical cuts: Text vs Display

A family whose name contains "Display" is not automatically a display face — many families ship multiple optical cuts of the same design. Freight, for instance, comes as Freight Text, Freight Display and Freight Big: the Text cut has sturdier strokes and a looser fit that survive paragraph sizes, while Big carries hairline contrast that only resolves at poster scale. Match the cut to the size you're actually setting, not to the vibe of its name.

## How many, and how to pair

- One face is a fine answer. Two covers nearly every product. Past three, hierarchy dissolves into noise. Marketing pages get more latitude than application UI.
- Weights and sizes obey the same economy: each exists to mark hierarchy, and stacking too many of them erases the very distinctions they were meant to draw.
- Pair across categories, not within them. A serif for display over a sans for text reads as a considered split between showing and reading. Two nearly interchangeable grotesques read as an oversight.

## Scope: a fix is not a redesign

No typography fix requires a new typeface. Smoothing, wrap modes, tabular figures, underline tuning — all of it operates inside whatever family the project already uses. Only swap or add a face when the task explicitly asks for one, and never pull in a commercial font to tick a review box.

When a face change *is* the task, two honest options:

```css
/* Native feel: hand rendering to the operating system */
body {
  font-family: system-ui, sans-serif;
}

/* Licensed brand face, backed by fallbacks with similar proportions */
body {
  font-family: "Graphik", "Helvetica Neue", Arial, sans-serif;
}
```

A branded face is a business decision as much as a visual one; whatever is chosen, the stack behind it must degrade gracefully.

## Why equal sizes look unequal

`font-size` sets the height of the em box, not of any visible letter. Within that box, every design makes its own choices: the **x-height** (how tall a lowercase x stands), the **cap height** (how tall the capitals stand), where the **baseline** sits, and how far **ascenders** reach above the x-height and **descenders** hang below the baseline. A face with a generous x-height fills more of its box and simply reads bigger than a small-x-height face at the same declared size — which is why swapping families often means re-tuning sizes, and why fallback fonts can jolt the layout when they render first.
