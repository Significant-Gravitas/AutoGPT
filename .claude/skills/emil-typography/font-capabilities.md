# Font capabilities: variable axes and OpenType

Modern font files do far more than draw one set of letters. This file covers what's inside them and the CSS that unlocks it.

## One file, many styles

A **static** font binds a single weight and style to a single file — a family sold as Light, Regular, Semibold and Bold means four separate binaries. A **variable** font packs a continuous design space into one file, so any intermediate value is real: `font-weight: 473` renders an actual 473, not a rounding to the nearest file.

Neither wins by default. A project using exactly one or two weights often ships fewer bytes with static files. A project that spans many weights, an optical-size axis or bespoke axes almost always comes out ahead with a variable file.

## Axes

Each axis of variation carries a four-letter tag. Five tags are registered standards; everything else is up to the type designer, and a given font exposes only the axes it was built with.

| Tag | Axis | What it varies |
| --- | --- | --- |
| `wght` | Weight | Stroke heaviness, the territory of `font-weight` |
| `wdth` | Width | How condensed or extended the glyphs are |
| `opsz` | Optical size | Contrast, spacing and detail retuned per display size |
| `slnt` | Slant | Oblique angle |
| `ital` | Italic | Toggles true italic forms |
| custom | — | Anything the designer invents, in uppercase tags |

Fraunces is a good tour: alongside `wght` and `opsz` it exposes `SOFT` (rounds the corners) and `WONK` (swaps in deliberately irregular alternates). A leaner file like Manrope's variable build exposes `wght` alone.

Optical sizing is older than variable fonts. Plenty of families still deliver it as separate cuts — see the Freight example in [typefaces-and-formats.md](typefaces-and-formats.md) — and a variable font with `opsz` handles it automatically when `font-optical-sizing: auto` is on.

## The property beats the raw tag

Whenever CSS offers a dedicated property, it must win over `font-variation-settings`. The property degrades correctly — `font-weight: 620` still bolds a fallback font — whereas a raw `"wght"` tag addressed at a font that never loaded does exactly nothing. Keep the low-level syntax for custom axes only:

```css
/* Right: registered axes go through their properties */
.masthead {
  font-weight: 620;
  font-optical-sizing: auto;
}

/* Right: a custom axis has no property, so the tag is the only door */
.pull-quote {
  font-variation-settings: "SOFT" 80;
}

/* Wrong: fallback fonts ignore this entirely */
.masthead {
  font-variation-settings: "wght" 620;
}
```

## Never let the browser improvise

Request an unloaded weight or style and the browser will manufacture one — mechanically thickened strokes for fake bold, sheared glyphs for fake italic. Useful as a last resort, terrible as a silent default. Turn it off so a missing file announces itself:

```css
html {
  font-synthesis: none;
}
```

## OpenType features

OpenType — the format underneath virtually every production font — lets a designer bundle optional behaviors called features. Unlike axes, features behave identically in static and variable fonts. As always, a font offers only what its designer put in. Frequently useful tags:

- `tnum` — tabular figures; every digit occupies the same width
- `lnum` / `onum` — lining versus old-style (lowercase-like) figures
- `zero` — slashed zero, so `0` can't impersonate `O`
- `frac` — proper diagonal fractions from sequences like 1/2
- `liga` — standard ligatures, merging pairs such as fi
- `ss01`–`ss20` — stylistic sets, numbered grab-bags of alternate glyphs
- `cv01`–`cv99` — character variants, per-glyph alternates

The property-over-tag rule applies here too: `font-variant-*` properties first, `font-feature-settings` only for tags without one.

```css
/* Right: dedicated properties exist for these */
.countdown {
  font-variant-numeric: tabular-nums;
}

.serial {
  font-variant-numeric: slashed-zero;
}

/* Right: numbered sets have no property */
.wordmark {
  font-feature-settings: "ss02" 1;
}
```

Tabular figures deserve special emphasis: any value that updates in place — timers, live prices, counters — will shove its neighbors around on every change unless digits share one width.

## Small caps, superiors, inferiors

- **Small caps** are purpose-drawn capital forms at lowercase scale — request the real ones with `font-variant-caps: small-caps` rather than shrinking uppercase text.
- **Superior figures** ride above the line (the ² in m²); **inferior figures** hang below (the ₂ in CO₂). `font-variant-position: super` / `sub` selects genuine glyphs instead of scaled-and-shifted imposters.

Both silently require the glyphs to exist in the font; verify before relying on them.

## Reading the numbered slots

Stylistic sets and character variants are numbered precisely because their contents are arbitrary: `ss01` in one family restyles the digits, in another it flattens terminals, in a third it does something else again. The number is a slot, not a meaning. Always consult the foundry's specimen or feature documentation before enabling one, and leave a comment saying what the slot does in this particular font — the next reader cannot guess.
