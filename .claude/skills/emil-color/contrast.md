# Contrast & Accessibility

## First, find the real pair

A contrast number only means something for a specific pairing: the text/icon/control color versus the surface it actually paints over — usually whatever ancestor supplies the nearest opaque background. Establish that pairing before measuring anything; checking a foreground against the wrong surface produces confidently wrong conclusions.

## The one-channel repair

Legacy workflows fix contrast by fiddling with all of R, G, and B (or worse, HSL saturation) until a checker turns green. In oklch the problem is one-dimensional: perceived contrast tracks the lightness gap between the pair, while chroma and hue contribute almost nothing. So the repair is always the same move — push the foreground's L away from the background's L, leaving C and H alone so the color keeps its identity.

```css
/* fails: lavender text nearly as light as its lavender card */
.card       { background: oklch(0.93 0.03 300); }
.card-label { color: oklch(0.72 0.14 300); }

/* passes: only L moved; the hue and vividness survive */
.card-label { color: oklch(0.42 0.14 300); }
```

Raising C to "make it pop" is the classic dead end — vividness is not contrast.

## Measuring with APCA

Prefer APCA (Accessible Perceptual Contrast Algorithm) for design decisions. It models perceived lightness difference — the same foundation oklch is built on — where WCAG 2's ratio formula misjudges many pairs in both directions.

APCA reports Lc, a signed score: dark-on-light pairs come out negative, light-on-dark positive. Judge the magnitude, ignoring sign. Working minimums:

- Body-size text: |Lc| ≥ 60, with 75+ as the comfortable target
- Large or heavy text: |Lc| ≥ 45, with 60+ preferred
- Non-text essentials (borders, icons, focus rings): |Lc| ≥ 30

Treat these as floors rather than goals.

## Measuring with WCAG 2

When a contract, audit, or policy names WCAG 2.x, its luminance-ratio math is what you must satisfy, whatever APCA says:

- Body text: at least 4.5:1 for level AA; 7:1 for AAA
- Large text — 18px+, or bold at 14px+: 3:1 for AA; 4.5:1 for AAA
- UI components and meaningful graphics: 3:1

Know its limitation: the formula is blunt, flagging some perfectly readable pairs while passing some murky ones. For compliance work, satisfy both systems; the union is rarely much stricter than APCA alone.

## Heuristics for fast triage

Estimating before computing:

- **Is the surface light or dark?** L above 0.6 → treat as light, pair with dark content; otherwise pair with light content.
- **Near-white surface (L ≥ 0.85):** foregrounds want L at or under 0.45.
- **Near-black surface (L ≤ 0.25):** foregrounds want L at or over 0.75.

These get you within range in one guess. They don't replace running the actual APCA or WCAG calculation before shipping — borderline cases and mid-tone backgrounds (L between 0.3 and 0.6, where no foreground has much room) need real numbers.
