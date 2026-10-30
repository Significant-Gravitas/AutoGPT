# Scale and rhythm

Size, leading and tracking carry more of typography's quality than any decorative flourish. Get these three right and most text problems never appear.

## Units for text

- `rem` — relative to the root font size; the default choice for font sizes because it honors the reader's browser settings.
- `em` — relative to the current element's font size; handy for things that should ride along with the text, like icon sizing or underline offsets.
- `%` on `font-size` — behaves like `em`, relative to the parent.
- `px` — frozen; ignores user font-size preferences. Acceptable for hairlines and borders, suspicious on text.

## Build a scale, then obey it

A type scale is a short, closed list of sizes that the whole product draws from. Its value is the constraint: when every size must come from the list, hierarchy stays consistent by construction, and the alternative — ad-hoc values sprinkled per component — reliably rots as the codebase grows.

A ratio-derived example (major third-ish, 1.2, on a 16px root):

```css
:root {
  --step--1: 0.833rem;
  --step-0: 1rem;
  --step-1: 1.2rem;
  --step-2: 1.44rem;
  --step-3: 1.728rem;
  --step-4: 2.074rem;
}
```

Ready-made scales are equally legitimate — Tailwind's `text-xs` through `text-9xl` pairs every size with a tuned default line-height and needs no setup.

### Name steps by role on teams

`text-lg` describes a size; nothing stops a teammate from using it anywhere. `text-card-title` or `text-caption` describes a job, and the name polices its own usage. On a solo project, size-based names plus written conventions are fine; the moment several people commit, role-based names pay for themselves.

## Leading (line-height)

Two anchors:

- **Body copy:** `1.5`–`1.6`. Reading rhythm needs air between lines.
- **Display and headings:** around `1.1`. Large sizes carry their own optical spacing; default leading leaves them gappy.

State it unitless. `line-height: 1.5` recomputes when the font size changes; `line-height: 24px` is a time bomb that detonates on the first size tweak or user zoom. Tailwind's `leading-snug` / `leading-normal` / `leading-relaxed` express the same idea and rarely need overrides.

## Tracking and kerning

These are different mechanisms and only one of them is yours to set routinely:

- **Kerning** is the font's built-in adjustment for specific awkward pairs — the classic gaps in "To" or "Wa". Browsers apply it automatically from the font's tables. `font-kerning: none` exists, but disabling kerning is a deliberate, unusual act.
- **Tracking** (`letter-spacing`) adds uniform space between all letters, and the right amount depends on size and case:
  - Display sizes: slightly negative — large glyphs sit too far apart at their default fit.
  - Small uppercase labels: slightly positive — capitals need daylight to stay distinguishable at caption sizes.
  - Body text: zero. Reading sizes are fit for reading already.

```css
.hero-title {
  font-size: var(--step-4);
  letter-spacing: -0.015em;
}

.eyebrow {
  font-size: var(--step--1);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
```

Track in `em` so the adjustment scales with the text it belongs to.

## Cropping built-in space with text-box

Every font reserves headroom above the cap height and legroom below the baseline. That reserve is why a label never sits dead-center in its button and why a heading's visual top doesn't touch its container. The `text-box` property crops it, taking an edge selector — `trim-both`, `trim-start`, `trim-end` — plus where each trimmed edge should land:

- `cap` — the cap height, for a tight top edge
- `alphabetic` — the baseline, for a tight bottom edge
- `text` — the font's own text edges, preserving descender room

```css
/* Center the label optically, not just geometrically */
.tag {
  text-box: trim-both cap alphabetic;
}

/* Pull a section title flush against the block above it */
.section-title {
  text-box: trim-start cap;
}

/* Snug bottom edge only */
.stat-label {
  text-box: trim-end alphabetic;
}
```

Support hasn't reached everywhere yet, so ship it as progressive enhancement: layouts must remain acceptable where the trim never happens.
