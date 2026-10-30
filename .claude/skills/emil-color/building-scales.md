# Building Tonal Scales & Palettes

## Step naming

Design systems label a hue's tonal steps numerically, lightest to darkest: 50, 100, 200 … 900, 950. The full eleven-step ramp is what Tailwind ships; leaner systems keep the shape but drop stops — a nine-step version omits 400 and 600, and a minimal five-step version keeps just 100/300/500/700/900. Whatever the count, 50 is always the near-white tint and 950 the near-black shade.

## Constructing a ramp

A good oklch ramp comes from three decisions, applied per step:

**1. Anchor the lightness endpoints, then space evenly.** Put step 50 around L 0.97 and step 950 around L 0.22, and interpolate linearly between them. Even L spacing is the point of using oklch — every adjacent pair of steps will look equally far apart. Don't run all the way to L 0 or L 1: at the poles chroma collapses to zero and the endpoints stop reading as members of the hue.

**2. Shape chroma as a curve, not a constant.** Real-world ramps are most vivid in the middle and gentle at both ends. Express each step's chroma as a fraction of that step's sRGB ceiling — e.g. ~30% at the extremes rising to ~85% around the 400–500 range — so the tints stay airy, the shades stay rich, and nothing leaves the gamut.

**3. Clamp against the gamut per step.** The ceiling `maxC(L, H)` differs at every lightness, so compute (or look up) it for each step before applying the fraction. A base color that is highly saturated will necessarily lose absolute chroma toward both ends of the ramp — that's the geometry of the gamut, not an error.

Hue stays fixed for all eleven steps.

Applied to a magenta-pink brand hue (H = 330):

```css
:root {
  --berry-50:  oklch(0.97 0.007 330);
  --berry-100: oklch(0.895 0.042 330);
  --berry-200: oklch(0.82 0.103 330);
  --berry-300: oklch(0.745 0.189 330);
  --berry-400: oklch(0.67 0.251 330);
  --berry-500: oklch(0.595 0.231 330);
  --berry-600: oklch(0.52 0.195 330);
  --berry-700: oklch(0.445 0.151 330);
  --berry-800: oklch(0.37 0.105 330);
  --berry-900: oklch(0.295 0.063 330);
  --berry-950: oklch(0.22 0.03 330);
}
```

Note the absolute chroma peaking near 400 rather than 500 — at this hue the sRGB gamut is widest around L 0.67, and the clamp reflects that.

## Families of hues

To make several hues feel like one system (the classic "brand + success + warning" trio), equalize two things across them:

- **The same L per step number** — every `-500` sits at the same lightness, so rows of the palette look equally bright.
- **The same fraction of each hue's own ceiling** — *not* the same absolute C. Ceilings differ wildly between hues, so a shared absolute value over-saturates the roomy hues and flattens the cramped ones.

```css
:root {
  /* all at L 0.65, each at 70% of its own sRGB ceiling */
  --teal-500:   oklch(0.65 0.08 185);   /* ceiling here ≈ 0.115 */
  --amber-500:  oklch(0.65 0.096 75);   /* ceiling here ≈ 0.137 */
  --violet-500: oklch(0.65 0.155 300);  /* ceiling here ≈ 0.221 */
}
```

Three different absolute chromas, one perceived intensity.

## Dark mode by inversion

Never hand-pick a second palette. Instead, keep one scale and swap which steps the semantic roles point at:

```css
:root {
  --surface: var(--berry-50);
  --ink: var(--berry-950);
}

[data-theme="dark"] {
  --surface: var(--berry-950);
  --ink: var(--berry-50);
}
```

The symmetry holds because oklch lightness is perceptually linear: a pair of steps separated by ΔL reads just as distinct after the flip as before it. Mid-scale steps (borders, muted text) swap with their mirror counterparts the same way — 200 ↔ 800, 300 ↔ 700.

## The case against HSL ramps

Two structural defects make HSL unusable for scale generation:

**The hue lies as lightness moves.** Take an HSL blue ramp at a fixed hue of 240: `hsl(240 65% 25%)` lands at oklch hue ≈ 272.6, `hsl(240 65% 55%)` at ≈ 274.3, but the pale `hsl(240 65% 88%)` tint lands at ≈ 285.2 — a ~13° slide toward violet that users see as the light steps "going purple." An oklch ramp pins H and the slide disappears.

**The lightness number is decorative.** `hsl(55 100% 50%)` (a yellow) and `hsl(255 100% 50%)` (a blue-violet) claim identical lightness; on screen one is nearly white-bright and the other is dark. Any ramp spacing computed from HSL's L channel inherits this distortion. OKLCH's L is the actual perceived brightness, so spacing computed from it is real.

## Auditing an inherited scale for drift

1. Convert every step of the existing scale to oklch.
2. Read off the H values and find the spread between the minimum and maximum.
3. Over ~10° of spread is visible drift: keep the scale's mid-tone hue, then regenerate the ramp with the procedure above.
