---
name: emil-surfaces
description: "Shadows, borders, gradients, depth, and dark-mode surface treatment for interfaces. Use when styling cards, buttons, containers, modals, dropdowns, or images, choosing between borders and shadows, fixing nested border radii, adding gradients or fades, defining elevation, or adapting surfaces to dark mode. Triggers on: box-shadow, shadow, border, borders, border-radius, rounded corners, nested radius, concentric radius, elevation, depth, layering, surface, card, container, gradient, gradient banding, eased gradient, mask-image, fade, fade-out, hairline border, 0.5px border, divider, outline, image outline, image border, dark mode, dark theme, theme variables, color scale, oklch, \"looks flat\", \"border looks harsh\", \"card feels off\"."
---

# Surfaces

Depth in an interface comes from how surfaces meet: the radius where they nest, the shadow or border that separates them, the gradient that fades them out. Most interfaces that feel "off" fail here — mismatched nested radii, harsh solid borders, banded gradients, or light-mode shadows pasted into dark mode. These are cheap to fix and compound into an interface that feels considered.

## Match the project's styling system

Before making changes, check how the project styles things — Tailwind, plain CSS, CSS Modules, or CSS-in-JS — and express every fix in that system. Don't introduce a `styles.css` full of custom classes into a Tailwind codebase, and don't inline Tailwind-style utilities into a styled-components project. Shadow and color tokens belong wherever the project keeps its design tokens (Tailwind `@theme` / config, `:root` variables, or the theme object).

## Core Principles

### 1. Derive nested radii, don't repeat them

A rounded child inside a rounded parent gets its radius by subtraction: `innerRadius = outerRadius − padding`. Copying the parent's radius onto the child makes the corner gap pinch — the single most frequent giveaway of an unpolished UI. The rule loses force once the gap between layers passes `24px`; at that distance the surfaces read as independent, so pick each radius on its own merits.

### 2. Shadows over borders for depth

Elements that use a border to signal elevation — cards, buttons, containers — should carry a stacked translucent `box-shadow` instead. Transparency composites with whatever sits underneath, so one token works everywhere; a solid border color is tuned to exactly one background. Keep real borders where the job is layout separation (dividers, table cells, form inputs).

### 3. Dark mode flattens shadows to rings

Stacked depth shadows can't be seen against dark surfaces. When the theme goes dark, collapse each shadow token down to one low-opacity white ring at 1px.

### 4. Hairline borders on retina

Dividers and fine lines should render at `0.5px` on high-density screens via a media-query-driven CSS variable, falling back to `1px` on standard displays.

### 5. Eased gradients, masks for fades

Plain linear gradients between solid colors band visibly — use eased gradients instead. For fading content out, prefer `mask-image` over a gradient overlay; masks work with whatever is behind them. Never put a fade on scrollable content — it cuts off what the user is trying to read.

### 6. Image outlines: pure black or pure white only

Every image gets a `1px` low-opacity outline for consistent depth. Light mode: black at 10%. Dark mode: white at 10%. Never a tinted neutral (slate, zinc, `#111827`) and never the accent color — a tinted line inherits whatever surface sits behind it and ends up looking like grime along the edge instead of a clean boundary.

### 7. Dark mode flips variables, not classes

Define colors as a numbered scale of CSS variables and swap the values under the dark theme selector. Don't scatter manual `dark:` overrides across components.

## Nested Radii

The inner radius is computed, never copied:

```css
/* Derived — corner gap stays even all the way around */
.dialog {
  border-radius: 16px;
  padding: 6px;
}
.dialog-body {
  border-radius: 10px; /* 16 − 6 */
}

/* Pinched — child repeats the parent's radius */
.dialog,
.dialog-body {
  border-radius: 16px;
}
```

```tsx
// Tailwind — subtract the padding when picking the child's radius
<section className="rounded-3xl p-3">   {/* 24px outer, 12px gap */}
  <article className="rounded-xl">      {/* 24 − 12 = 12px ✓ */}
    ...
  </article>
</section>
```

## Depth Without Borders

A card or button that draws a solid border to look raised has two problems: the color was picked against one specific background, and the edge reads flat. Replace it with a shadow stack built from three ingredients — a 1px spread that stands in for the border, a tight crisp layer just beneath the element, and a wider soft layer for ambient depth:

```css
:root {
  --elevation-raised:
    0 0 0 1px rgb(0 0 0 / 0.05),
    0 1px 3px rgb(0 0 0 / 0.05),
    0 4px 10px -4px rgb(0 0 0 / 0.05);
  --elevation-raised-hover:
    0 0 0 1px rgb(0 0 0 / 0.07),
    0 1px 3px rgb(0 0 0 / 0.07),
    0 4px 10px -4px rgb(0 0 0 / 0.08);
}

/* Hook into however the project switches themes:
   a class, a data attribute, or prefers-color-scheme */
.dark {
  --elevation-raised: 0 0 0 1px rgb(255 255 255 / 0.09);
  --elevation-raised-hover: 0 0 0 1px rgb(255 255 255 / 0.14);
}
```

For hover, transition only the shadow — nothing else:

```css
.panel {
  box-shadow: var(--elevation-raised);
  transition: box-shadow 150ms ease-out;
}
.panel:hover {
  box-shadow: var(--elevation-raised-hover);
}
```

The lone spread ring is also useful on its own, anywhere a real hairline border sits awkwardly against its background:

```css
/* Ring shadow standing in for a 1px border */
box-shadow: 0 0 0 1px rgb(0 0 0 / 0.07);
```

Leave separation borders alone. A bottom border between list rows, a table's gridlines, or an input's visible edge is doing layout work, not depth work — converting those to shadows gains nothing, and on form fields it costs accessibility.

### Picking between the two

**Reach for a shadow** when the element reads as raised: cards, bordered button styles, menus, popovers, dialogs, anything that travels across mixed or image backgrounds, and hover/focus lift effects.

**Keep a border** when the line separates rather than elevates: row dividers, table gridlines, form-field edges (a visible outline is an accessibility feature), and hairline rules in dense layouts.

## Hairline Borders

For the borders you keep, use density-aware hairlines so dividers stay crisp on retina screens:

```css
:root {
  --border-hairline: 1px;

  @media only screen and (min-device-pixel-ratio: 2),
    only screen and (min-resolution: 192dpi) {
    --border-hairline: 0.5px;
  }
}

.divider {
  border-bottom: var(--border-hairline) solid var(--gray-6);
}
```

## Gradients and Fades

**Eased over linear.** Linear gradients between solid colors show banding. Use an eased gradient (multiple stops following an easing curve — generate with https://larsenwork.com/easing-gradients/).

**Mask over gradient overlay.** To fade content out, prefer `mask-image` — it composites with any content or background, where a gradient overlay only matches one background color:

```css
.fade-bottom {
  mask-image: linear-gradient(to bottom, black 80%, transparent);
}
```

**No fades on scrollable content.** A fade over a scrollable list restricts the viewable area and cuts off content the user is scrolling to read.

## Image Outlines

Photos and screenshots should carry the same edge treatment as every other surface: a 1px translucent line drawn just inside the image bounds.

```css
img {
  outline: 1px solid rgb(0 0 0 / 0.1);
  outline-offset: -1px; /* pulled inside — costs no layout space */
}

.dark img {
  outline-color: rgb(255 255 255 / 0.1);
}
```

```tsx
// Tailwind
<img
  alt={alt}
  src={src}
  className="-outline-offset-1 outline-1 outline-black/10 dark:outline-white/10"
/>
```

The color is fixed and not up for theming: black at 10% in light mode, white at 10% in dark. Palette neutrals (`outline-slate-*`, `outline-zinc-*`, `outline-neutral-*`) and accent colors are off the table — anything tinted picks up the surface color behind the image and looks like grime rather than a clean edge. Draw it with `outline` instead of `border` (no effect on box size), and inset it with `outline-offset: -1px` so the image keeps its intended dimensions.

## Dark Mode Surfaces

Use a numbered color scale as CSS variables and flip the values per theme:

```css
:root {
  --gray-1: #fafafa;
  --gray-2: #f5f5f5;
  --gray-12: #171717;
}

[data-theme="dark"] {
  --gray-1: #171717;
  --gray-2: #1f1f1f;
  --gray-12: #fafafa;
}
```

In Tailwind projects, do not hand-write `dark:` color overrides per component. Point utilities at the variables and let the theme flip do the work:

```css
/* Good — variables flip automatically */
.button {
  background: var(--gray-12);
  color: var(--gray-1);
}

/* Avoid — manual overrides everywhere */
.button {
  @apply bg-gray-900 dark:bg-gray-100;
}
```

And remember the two dark-mode surface rules from above: shadows collapse to a single white ring, and image outlines switch to pure white at 10%.

## Symptoms and Their Fixes

- **Corners pinch where a rounded child meets its parent** — the child copied the radius; derive it instead (outer minus padding).
- **A card looks stamped onto the page rather than raised** — solid depth border; swap in the layered shadow token.
- **Cards disappear in dark mode** — the light shadow stack is still active; collapse it to the white ring.
- **Shadow hover feels laggy or heavy** — the element transitions `all`; scope the transition to `box-shadow` only.
- **Visible stripes across a large gradient** — banding; add eased intermediate stops.
- **A fade only looks right on one background** — it's a gradient overlay; rebuild it with `mask-image`.
- **A fade sits over a scrolling list** — remove it; it hides exactly what the user is scrolling toward.
- **Image edges look dirty or smudged** — tinted outline; switch to pure black/white at 10%.
- **`dark:` overrides sprinkled through every component** — flip a numbered variable scale at the theme root instead.

## Review Checklist

- [ ] Inner radii derived (outer − padding) wherever rounded surfaces nest
- [ ] Cards, buttons, and containers signal depth with shadow tokens, not solid borders
- [ ] Separation borders left intact — and hairline-thin on high-DPI screens
- [ ] Dark theme collapses every shadow stack to a single white ring
- [ ] Large gradients eased; fades built with `mask-image`; nothing fading over a scroll area
- [ ] Every image carries the 1px inset outline — pure black/white at 10%, never tinted
- [ ] Themes switch by flipping variables, not per-component `dark:` classes

For non-surface finishing details — font rendering, layout shift, focus and hover states, z-index, scrollbars — see the `emil-ui-polish` skill.
