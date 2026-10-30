# Gamut, Display P3 & Tailwind v4

## The shape of the displayable

OKLCH describes more colors than any screen can show. Whether a given value is renderable depends on the target gamut: sRGB is the safe universal floor, while Display P3 — standard on modern phones, Macs, and better monitors — is a strict superset of sRGB, roughly a quarter larger, with its extra room concentrated in intense greens, reds, and pinks.

Within either gamut, the chroma ceiling swings dramatically with both hue and lightness. Some computed sRGB reference points:

- At mid lightness (L 0.5), violet-purples around H 285 enjoy the most headroom (C up to ≈ 0.29), reds around H 25 manage ≈ 0.20, and cyans near H 195 are pinned under ≈ 0.09 — a 3× spread between the roomiest and tightest hues.
- The roomy region migrates as lightness changes: by L 0.7 the peak sits in the pinks (H ≈ 330, C ≈ 0.31), and by L 0.9 it has moved to spring greens (H ≈ 130, C ≈ 0.24).
- Cyan is cramped at every lightness — plan for muted teals in sRGB.

## When a value doesn't fit

A color whose C exceeds the ceiling for its L and H gets gamut-mapped by the browser — with results you didn't choose. Handle it yourself instead: keep L and H (they carry the color's identity) and pull C down to the boundary.

```css
/* asks for more red than sRGB has at this lightness */
color: oklch(0.6 0.31 20);

/* largest C that renders faithfully: */
color: oklch(0.6 0.241 20);
```

## Progressive enhancement for P3

Ship the sRGB-safe value unconditionally and layer the wider-gamut version behind a capability query:

```css
.cta {
  background: oklch(0.66 0.2 330);
}

@media (color-gamut: p3) {
  .cta {
    background: oklch(0.66 0.32 330);
  }
}
```

Here the sRGB chroma ceiling at this lightness and hue is about 0.30, and P3's is about 0.33 — so 0.32 renders only on wide-gamut screens, and everyone else keeps the safe base.

If the project still supports pre-2023 browsers with no `oklch()` parser at all, put a hex floor underneath and gate the modern syntax with `@supports`:

```css
.cta {
  background: #d25bcb;
}

@supports (color: oklch(1 0 0)) {
  .cta {
    background: oklch(0.66 0.2 330);
  }

  @media (color-gamut: p3) {
    .cta {
      background: oklch(0.66 0.32 330);
    }
  }
}
```

## Tailwind v4

From v4, Tailwind's bundled palette is authored in oklch, and the framework expects custom theme colors in the same notation.

### Declaring a scale

Tokens named `--color-<name>-<step>` inside `@theme` automatically materialize the full utility set — `bg-berry-400`, `text-berry-100`, `border-berry-700`, and friends:

```css
@theme {
  --color-berry-50:  oklch(0.97 0.007 330);
  --color-berry-100: oklch(0.895 0.042 330);
  --color-berry-200: oklch(0.82 0.103 330);
  --color-berry-300: oklch(0.745 0.189 330);
  --color-berry-400: oklch(0.67 0.251 330);
  --color-berry-500: oklch(0.595 0.231 330);
  --color-berry-600: oklch(0.52 0.195 330);
  --color-berry-700: oklch(0.445 0.151 330);
  --color-berry-800: oklch(0.37 0.105 330);
  --color-berry-900: oklch(0.295 0.063 330);
  --color-berry-950: oklch(0.22 0.03 330);
}
```

(Scale construction — lightness spacing, chroma curve, per-step clamping — is covered in [building-scales.md](building-scales.md).)

### Opacity suffixes

The slash modifier composes transparently with oklch tokens: `text-berry-600/75` emits `oklch(0.52 0.195 330 / 0.75)`. No separate alpha variables needed.

### Porting a hex-based theme

1. Re-express every `@theme` color token as oklch.
2. Chase down `theme()` calls and component-level hardcoded hex that duplicated those tokens, and point them at the converted values.
3. Re-run your contrast checks and eyeball dark mode: exact conversion preserves each color, but colors that were *approximated* by eye in hex often turn out slightly off once measured perceptually.
4. Where the brand wants more punch than sRGB allows, add the `@media (color-gamut: p3)` layer from above rather than raising the base token's chroma.
