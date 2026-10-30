---
name: emil-ui-polish
description: "The invisible details that make interfaces feel finished — font rendering, layout-shift prevention, interaction states, hit areas, and structural hygiene. Use when building or reviewing UI components, doing a polish pass, fixing \"something feels off/unfinished\", or working on text rendering, hover/focus/selection states, scrollbars, stacking, truncation, or empty states. Triggers on: polish, details, font-smoothing, antialiased, font rendering, tabular-nums, tabular numbers, layout shift, CLS, jumping text, skeleton, placeholder, scrollbar, z-index, stacking context, isolation, focus ring, focus outline, ::selection, hover state, active state, pressed state, scale on press, hit area, tap target, touch target, optical alignment, icon centering, truncation, line-clamp, ellipsis, curly quotes, text-wrap balance, safe-area, scroll-margin, empty state, pointer-events, user-select, flash on refresh, \"feels off\", \"feels unfinished\", \"make it feel better\"."
---

# UI Polish

A finished interface is mostly invisible work: text that renders crisply, numbers that don't jiggle, hover states that don't shift layout, focus rings that don't clash, hit areas you never miss. No single one of these is noticeable on its own — their absence is. Apply them as a pass over every component you build or review.

## Match the project's styling system

Check how the project styles things — Tailwind, plain CSS, CSS Modules, or CSS-in-JS — and express every fix in that system. Don't add a global stylesheet to a Tailwind codebase or utility classes to a styled-components one. Tokens (z-index scale, font weights) go wherever the project keeps its design tokens.

## Core Principles

### 1. Antialiased font smoothing, always

```css
body {
  -webkit-font-smoothing: antialiased;
}
```

Also subset custom fonts to the characters, alphabets, and languages actually used — nothing else ships.

### 2. Never change font weight on interaction

Weight changes on hover or selection reflow the text and shift layout. Keep weight constant and signal state with color:

```css
/* Bad — layout shift */
.tab:hover { font-weight: 600; }

/* Good — constant weight, color signals state */
.tab { font-weight: 500; }
.tab.selected { color: var(--color-primary); }
```

Define weights as variables (`--font-weight-normal: 400; --font-weight-medium: 500; --font-weight-semibold: 600; --font-weight-bold: 700;`) so they're adjustable globally.

### 3. Tabular numbers for anything that changes

Counters, prices, timers — any dynamic number gets `font-variant-numeric: tabular-nums` so digits occupy equal width and don't jiggle as they update.

### 4. No layout shift from dynamic content

Skeleton loaders, image placeholders, and dynamic content areas get hardcoded dimensions (or `aspect-ratio`) matching the loaded state. Images always declare their dimensions so the page doesn't reflow as they arrive. For decorative image depth (outlines), see the `emil-surfaces` skill.

### 5. Real typographic characters

Use `…` not `...`, `'` not `'`, `"` `"` not straight quotes. Add `text-wrap: balance` on `h1`–`h3` for better line breaks. Larger text wants tighter letter spacing, smaller text looser — pair each font size with its tracking (a `Text` component is the right home for this; exact values are font-dependent).

### 6. Interaction states that stay put

- **Hover:** transition only the properties that change (`transition-property: color, background-color, box-shadow`), around `150ms ease-out`. Never `transition: all`. Hover must not change layout (see #2).
- **Press:** `scale(0.96)` on active for tactile feedback. Always `0.96` — never below `0.95`, which feels exaggerated. Tailwind: `active:scale-[0.96] transition-transform`.
- **Focus:** don't recolor the default focus outline to anything other than grey, black, or white — custom colored outlines clash with the interface. Never remove it.
- **Selection:** if you style `::selection` at all, keep it a subtle tint; don't make selected text unreadable or brand-colored for its own sake.
- `will-change` only for `transform`, `opacity`, `filter`, only when you observe first-frame stutter — never `will-change: all`.

### 7. Fixed z-index scale

No arbitrary `z-index: 9999`. Define a scale and use only its steps:

```css
:root {
  --z-dropdown: 100;
  --z-modal: 200;
  --z-tooltip: 300;
  --z-toast: 400;
}
```

Better: avoid z-index entirely where possible — `isolation: isolate` (or `position: relative`) creates a local stacking context so children can't leak above unrelated UI.

### 8. Minimum hit areas

Interactive elements need `44×44px` hit areas for touch/mobile, at least `40×40px` on desktop. Extend small controls with a pseudo-element. Two hit areas must never overlap — if an extended area collides with a neighbor, shrink it to the largest non-colliding size.

### 9. Align by eye, not by math

Mathematical centering often reads as off-center; nudge elements until they look right:

- Button with trailing icon: icon-side padding = text-side padding − 2px (`pl-4 pr-3.5`).
- Play triangles: shift `2px` toward the pointed side (`margin-left: 2px`).
- Asymmetric icons (stars, arrows, carets): fix the SVG viewBox/path itself; a `1px` margin is the fallback.

### 10. Native scrollbars on the page

Never replace the page scrollbar. Customize scrollbars only inside small elements like code blocks — `8px` wide, `rgba(0,0,0,0.2)` thumb, `4px` radius.

### 11. Empty states teach

An empty list is a first-run screen, not an error. Show what belongs there and the action that creates it: short headline, one line of explanation, primary action. Size the empty state to the dimensions of the filled state so completing the action doesn't shift the layout.

### 12. Decorative elements are inert

Anything purely visual gets `pointer-events: none` so it can't steal clicks from real controls, and code-built illustrations get `user-select: none`.

### 13. No flash on refresh

Interactive components must render their real state on first paint: persist state to `localStorage`/`sessionStorage`, hydrate correctly under SSR, and set initial state before render — never default-then-correct.

## Details

### Hit area extension

Grow a small control's touchable region with a pseudo-element that bleeds past its box — negative `inset` is the simplest way:

```css
/* 16px dismiss icon, 40px effective target */
.toast-dismiss {
  position: relative;
}
.toast-dismiss::before {
  content: "";
  position: absolute;
  inset: -12px; /* 16 + 12 + 12 = 40 */
}
```

```tsx
// Tailwind
<button className="relative before:absolute before:-inset-3">
  <XIcon className="size-4" />
</button>
```

### Scoped scrollbar styling

```css
.code-block::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
.code-block::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.2);
  border-radius: 4px;
}
```

### Anchors, safe areas, truncation

```css
/* Room for the sticky header when scrolling to anchors */
[id] {
  scroll-margin-top: 80px; /* header height */
}

/* Notches and home indicators */
.footer { padding-bottom: env(safe-area-inset-bottom); }
.sidebar { padding-left: env(safe-area-inset-left); }

/* Truncate text in grid cells */
.grid-cell-text {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
```

## Frequent Offenders

| Problem | Remedy |
| --- | --- |
| Bold-on-hover tabs shifting layout | Constant weight; signal with color |
| Timer/price digits jiggling | `font-variant-numeric: tabular-nums` |
| Content jumping as images/skeletons load | Hardcode dimensions or `aspect-ratio` |
| `transition: all` | List exact properties |
| `scale(0.9)` on press | `scale(0.96)` — below `0.95` feels exaggerated |
| Brand-colored focus outline | Grey, black, or white only |
| `z-index: 9999` | Fixed scale, or `isolation: isolate` |
| 20px icon button with 20px hit area | Pseudo-element to 44×44 (40×40 desktop) |
| Icon visually off-center in button | −2px padding on icon side; fix SVG for asymmetric icons |
| Custom page scrollbar | Native on the page; customize only small scrollers |
| Blank div for empty list | Headline + explanation + action, sized like the filled state |
| Decorative layer blocking clicks | `pointer-events: none` |
| State flashing on refresh | Persist + set initial state before render |

## Review Checklist

- [ ] `-webkit-font-smoothing: antialiased` on body; fonts subsetted
- [ ] No font-weight changes on hover/selected states
- [ ] Dynamic numbers use `tabular-nums`
- [ ] Skeletons, placeholders, and images have fixed dimensions
- [ ] Proper typographic characters; `text-wrap: balance` on headings
- [ ] Transitions list specific properties; press uses `scale(0.96)`
- [ ] Focus outlines neutral and intact; `will-change` rare and specific
- [ ] Z-index from the scale, or stacking contexts via `isolation`
- [ ] Hit areas ≥ 44×44 touch / 40×40 desktop, never overlapping
- [ ] Icons optically centered
- [ ] Page scrollbar untouched; anchors have `scroll-margin-top`; safe areas padded
- [ ] Empty states designed; decorative elements inert; no refresh flash

Shadows, borders, gradients, elevation, and dark-mode surface treatment live in the `emil-surfaces` skill — use it alongside this one for visual depth work.
