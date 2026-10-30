---
name: emil-color
description: "Working with color in CSS through OKLCH. Use when a task involves picking, converting, or auditing colors — hex/rgb/hsl to oklch migration, building tonal scales and multi-hue palettes, deriving dark mode, diagnosing or repairing text contrast (APCA and WCAG), staying inside sRGB or Display P3 gamut, or defining Tailwind v4 theme tokens. Triggers on: oklch, oklab, color space, convert hex, rgb, hsl, palette, color scale, tonal ramp, shades, contrast checking, WCAG, APCA, a11y colors, readability, gamut, wide gamut, display-p3, srgb, chroma, hue, lightness, saturation, dark mode palette, theme colors, design tokens, @theme, Tailwind colors, color variables."
---

# Color

An `oklch()` value reads like a measurement, not a recipe. Lightness tells you how bright the color actually looks, chroma tells you how vivid it is, hue tells you where it sits on the wheel — and none of the three lies to you the way HSL's channels do. That honesty is the whole pitch: once the numbers track perception, palette math, contrast repair, and dark mode all become arithmetic instead of eyeballing. Default to oklch for anything new.

## Before touching any code

**Work inside the project's existing styling system.** Look at how colors are already declared — plain CSS custom properties, a Tailwind `@theme` block, CSS-in-JS theme objects, generated design tokens — and write your change in that same mechanism. A color fix is never a reason to bolt on a parallel theming layer.

## How to read a value

```css
color: oklch(0.586 0.222 17.585);          /* a saturated crimson */
color: oklch(0.398 0.066 227.392 / 0.9);   /* deep slate-cyan at 90% opacity */
```

The channels, in order:

- **L** runs 0 (black) to 1 (white) and is perceptually even — 0.5 genuinely looks mid-toned regardless of hue.
- **C** starts at 0 (pure gray) and tops out around 0.4 in extreme cases; how high it can actually go depends on both L and H, because the displayable gamut is lumpy.
- **H** is an angle, 0–360°. Rough landmarks: ~20 red, ~90 yellow, ~145 green, ~195 cyan, ~260 blue, ~330 pink.
- Opacity, when needed, goes after a slash — never as a fourth comma-separated argument.

Three decimals per channel is plenty of precision; trim trailing zeros and write `-0` as plain `0`. Support is a non-issue: every evergreen browser has shipped it since 2023 (it is Baseline), covering upward of 96% of users.

## Operating principles

**Everything new is oklch.** When editing code that already uses hex/`rgb()`/`hsl()`, swap the values and nothing else — structure, keywords like `currentColor` or `transparent`, and hex-expecting third-party config all stay put. Details in [converting-colors.md](converting-colors.md).

**L answers "light or dark."** A surface with L above 0.6 reads as light and wants dark text on it; at or below 0.6, reach for light text. No luminance formula needed for the first pass.

**Contrast lives in the L gap.** Chroma and hue barely move contrast; the distance in L between a foreground and the background it renders on is what counts. To repair a failing pair, move only L. Verify with APCA (|Lc| of 60 minimum for body text, 75 to be comfortable) or with WCAG 2 ratios (4.5:1 AA, 7:1 AAA) when compliance language is in play. Full guidance in [contrast.md](contrast.md).

**One hue per ramp.** Every step of a tonal scale keeps the identical H. If converting an old scale to oklch reveals more than about 10° of hue spread between its ends, the ramp drifts visibly — rebuild it. How in [building-scales.md](building-scales.md).

**Vividness is relative, not absolute.** Each hue has its own chroma ceiling at a given lightness, so two hues sharing one absolute C won't look equally saturated. For sibling colors, hold L constant and give each hue the same *fraction* of its own ceiling.

**Check the gamut before shipping high chroma.** Plenty of syntactically valid oklch values can't be shown on an sRGB screen. Clamp C (holding L and H) until the color fits, and serve P3-only chroma behind a `@media (color-gamut: p3)` gate on top of an sRGB base value. Patterns in [gamut-p3-tailwind.md](gamut-p3-tailwind.md).

**Dark mode is a remap, not a repaint.** Flip which end of the scale feeds which role — the token that was near-white becomes near-black and so on. Because L steps are perceptually even, the inverted assignments stay balanced without per-color tweaking.

**Tailwind v4 speaks oklch natively.** Its stock palette is authored in oklch, so custom `@theme` scales should be too — no hex tokens. The `/50`-style opacity suffix composes with oklch out of the box.

## Numbers worth memorizing

- Light-vs-dark surface cutoff: **L 0.6**
- Against a very light ground (L ≥ 0.85), text needs **L ≤ 0.45**; against a very dark ground (L ≤ 0.25), text needs **L ≥ 0.75** — heuristics, confirm with a real check
- Visible hue drift across a ramp: spread **> 10°**
- APCA body text: **|Lc| 60** minimum, **75** preferred; large text 45; non-text UI 30
- WCAG 2 body text: **4.5:1** (AA) and **7:1** (AAA); large text 3:1 and 4.5:1
- Contrast repairs touch **L only** — C and H stay

## Failure modes to catch

- A fresh hex or `hsl()` literal landing in new code → express it as oklch.
- A scale whose light and dark ends resolve to different perceptual hues → rebuild on a fixed H.
- Low-contrast text "fixed" by pumping saturation → does nothing; widen the L gap instead.
- One C value copied verbatim between hues ("all our 500s use 0.19") → uneven vividness; use per-hue percentages of the ceiling.
- Chroma beyond the sRGB ceiling with no fallback → clips unpredictably; clamp, then layer P3 via media query.
- A dark theme assembled from scratch-picked colors → derive it by inverting the light scale's L assignments.
- Hex values inside a Tailwind v4 `@theme` block → convert them.
- `oklch(0.5, 0.1, 200, 0.5)` comma-style alpha → invalid; alpha goes after `/`.

## Reporting changes

When a task ends with modified colors, walk through them grouped by file: quote each declaration's previous value beside its replacement and add a short clause naming the rule it violated (drifting hue, insufficient L gap, out-of-gamut chroma, …). Cover every declaration you touched, not just highlights — the reader should be able to audit the work line by line without re-deriving it.

## Going deeper

- [converting-colors.md](converting-colors.md) — mechanical hex/rgb/hsl → oklch migration, what to skip, whole-file passes
- [building-scales.md](building-scales.md) — tonal ramps, step naming, multi-hue families, dark mode derivation, why HSL ramps fail
- [contrast.md](contrast.md) — APCA and WCAG 2 in practice, repairing pairs via L, quick heuristics
- [gamut-p3-tailwind.md](gamut-p3-tailwind.md) — sRGB/P3 boundaries, clamping, progressive enhancement, Tailwind v4 theming
