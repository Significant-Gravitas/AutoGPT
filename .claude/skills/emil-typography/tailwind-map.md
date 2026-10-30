# Tailwind map

Every CSS declaration this skill mentions, translated into Tailwind (v4) so fixes can be written in whichever idiom the project uses. Where Tailwind ships no named utility, the bracketed arbitrary form is given — it compiles like any other utility.

## Family, weight, rendering

| Goal | CSS | Tailwind |
| --- | --- | --- |
| Italic style | `font-style: italic` | `italic` |
| Monospace stack | `font-family: monospace` | `font-mono` |
| Sans stack | `font-family: sans-serif` | `font-sans` |
| Serif stack | `font-family: serif` | `font-serif` |
| Size step from the scale | `font-size: …` | `text-*` |
| Weight (any of 1–1000) | `font-weight: …` | `font-*` |
| Block fake bold/italic | `font-synthesis: none` | `[font-synthesis:none]` |
| macOS rendering weight | `-webkit-font-smoothing` with `-moz-osx-font-smoothing` | `antialiased` |

## Variable-font and OpenType controls

| Goal | CSS | Tailwind |
| --- | --- | --- |
| Custom axis value | `font-variation-settings: "SOFT" 80` | `[font-variation-settings:"SOFT"_80]` |
| Size-aware detailing | `font-optical-sizing: auto` | `[font-optical-sizing:auto]` |
| Equal-width digits | `font-variant-numeric: tabular-nums` | `tabular-nums` |
| Distinguish 0 from O | `font-variant-numeric: slashed-zero` | `slashed-zero` |
| Genuine small caps | `font-variant-caps: small-caps` | `[font-variant-caps:small-caps]` |
| Genuine super/subscript glyphs | `font-variant-position: super` | `[font-variant-position:super]` |
| Numbered feature slot | `font-feature-settings: "ss02" 1` | `[font-feature-settings:"ss02"]` |
| Ligature control | `font-variant-ligatures: none` | `[font-variant-ligatures:none]` |

## Rhythm and fit

| Goal | CSS | Tailwind |
| --- | --- | --- |
| Leading | `line-height: …` | `leading-*` |
| Tracking | `letter-spacing: …` | `tracking-*` |
| Disable pair kerning | `font-kerning: none` | `[font-kerning:none]` |
| Reading-column cap | `max-width: 65ch` | `max-w-prose` or `max-w-[65ch]` |
| Crop leading above/below | `text-box: trim-both cap alphabetic` | `[text-box:trim-both_cap_alphabetic]` |
| Line start/end alignment | `text-align: …` | `text-start` / `text-center` |

## Breaking, overflow, casing

| Goal | CSS | Tailwind |
| --- | --- | --- |
| Even multi-line headings | `text-wrap: balance` | `text-balance` |
| No stranded final word | `text-wrap: pretty` | `text-pretty` |
| Forbid wrapping | `white-space: nowrap` | `whitespace-nowrap` |
| Allow long tokens to split | `overflow-wrap: break-word` | `break-words` |
| Dictionary hyphenation | `hyphens: auto` | `hyphens-auto` |
| One-line ellipsis | `text-overflow: ellipsis` (plus hidden overflow, nowrap) | `truncate` |
| N-line clamp | `line-clamp: 3` | `line-clamp-3` |
| Render-time casing | `text-transform: uppercase` | `uppercase` / `capitalize` |

## Links and decoration

| Goal | CSS | Tailwind |
| --- | --- | --- |
| Underline on | `text-decoration-line: underline` | `underline` |
| Underline tint | `text-decoration-color: …` | `decoration-*` |
| Underline weight | `text-decoration-thickness: …` | `decoration-1` / `decoration-2` |
| Weight from the font's metrics | `text-decoration-thickness: from-font` | `decoration-from-font` |
| Position from the font's metrics | `text-underline-position: from-font` | `[text-underline-position:from-font]` |
| Distance below baseline | `text-underline-offset: …` | `underline-offset-*` |
| Dotted/dashed/wavy line | `text-decoration-style: …` | `decoration-dotted` / `decoration-wavy` |
| Break line around descenders | `text-decoration-skip-ink: auto` | `[text-decoration-skip-ink:auto]` |

## Interaction and ornament

| Goal | CSS | Tailwind |
| --- | --- | --- |
| Insertion-bar color | `caret-color: …` | `caret-*` |
| Non-selectable text | `user-select: none` | `select-none` |
| Gradient-filled letters | `background-clip: text` | `bg-clip-text` |
| Outlined letters | `-webkit-text-stroke: 1px currentColor` | `[-webkit-text-stroke:1px_currentColor]` |
| Letterform shadow | `text-shadow: …` | `text-shadow-*` |
| Drop-cap sizing | `initial-letter: 3` | `[initial-letter:3]` |
