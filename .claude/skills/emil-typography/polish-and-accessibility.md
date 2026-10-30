# Polish and accessibility

The last few percent — links, selection, carets, ornament — and the hard floors underneath all of it.

## Underlines

Left alone, the browser invents an underline: its distance and weight are guesses, and the guesses frequently collide with descenders or come out thread-thin. The cleanest correction defers to the type designer, whose font carries its own underline metrics:

```css
a {
  text-decoration-thickness: from-font;
  text-underline-position: from-font;
}
```

Manual tuning is the alternative when the font's metrics don't suit the design. Size the parts in `em` so they scale with the text:

```css
.prose a {
  text-decoration-color: color-mix(in oklab, currentColor 40%, transparent);
  text-decoration-thickness: 0.08em;
  text-underline-offset: 0.16em;
  text-decoration-skip-ink: auto;
  transition: text-decoration-color 150ms ease;
}

.prose a:hover {
  text-decoration-color: currentColor;
}
```

`text-decoration-skip-ink: auto` breaks the line around descending strokes instead of striking through them.

The style needn't be a solid line. `text-decoration-style` offers dotted, dashed and wavy variants, and dotted has a settled meaning in interfaces — "there's more here": abbreviations, glossary terms, anything with a tooltip attached.

```css
abbr[title] {
  text-decoration: underline dotted;
}
```

One hard constraint on animation: of a native underline's parts, only the color transitions reliably. A link underline that grows, slides or thickens on hover must be its own element (border, pseudo-element, or background gradient) that you animate freely.

## Selection and highlights

- `::selection` restyles highlighted text — an easy, quiet brand touch. Whatever background/foreground pair you choose has to remain comfortably readable; a decorative selection nobody can read is worse than the default.
- `user-select: none` belongs on button labels and control chrome, where a blue smear of selection reads as malfunction. Audit what select-all actually grabs on the page — it should collect content, not interface.
- Products aiming to feel native (Electron apps, PWAs) usually invert the default: selection off globally, re-enabled only on genuinely copyable content.
- `::target-text` styles the passage a text-fragment link (`#:~:text=`) jumps to.
- The CSS Custom Highlight API paints arbitrary ranges — search hits, collaborative cursors — without wrapping them in elements.

## Placeholders and carets

- `::placeholder` restyles the ghost text in empty inputs. Keep it visibly lighter than entered text, but not so faint it fails contrast for the people who need it most.
- `caret-color` tints the insertion bar to match the brand. That single property is effectively the whole API: replacing the caret outright means rebuilding text editing, which is almost never a good trade.

## Mobile input zoom

Mobile Safari treats sub-`16px` input text as unreadable-while-typing and responds by zooming the entire page when the field gains focus. The fix is to give inputs `16px` on small viewports and shrink them, if desired, only at wider breakpoints:

```tsx
// 16px below the md breakpoint keeps Safari from zooming
<input type="search" className="text-base md:text-sm" />
```

The tempting shortcut — `maximum-scale=1` in the viewport meta — is a trap: Safari itself ignores the cap for pinch gestures, while Chrome on Android and others enforce it and thereby strip users of zoom, violating WCAG 1.4.4 (Resize Text). The responsive font size costs nothing and harms no one.

## Font smoothing

macOS renders type noticeably heavier than the same font on other platforms. Opting into antialiased/grayscale rendering brings it back toward the intended weight. Declare it once at the root — both vendor properties together, which is precisely what Tailwind's `antialiased` utility emits:

```css
body {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
```

Once, globally, at the layout root — not re-declared per component.

## Ornamental text

- `::first-letter` — the classic drop-cap hook; support is universal.
- `initial-letter` — sizes the drop cap in line units; still missing from Firefox, so treat as enhancement.
- `::first-line` — styles exactly the first rendered line, reflowing as the container resizes.
- `background-clip: text` — masks a gradient or image to the glyph shapes (pair with transparent text color).
- `-webkit-text-stroke` — outlines glyph contours; despite the prefix it works in all modern engines.
- `text-shadow` — a shadow that follows letterforms rather than the box.

A known artifact: stroked text sometimes shows seams *inside* the letters. Those seams are overlapping contours in the font itself — variable fonts commonly leave shape intersections unmerged — and no CSS will remove them; a static instance of the font will.

## The floors

Everything above is negotiable styling; these are not. And remember that readers change text: zoom, enlarged default font size, forced line spacing. A layout that only works at one size is already broken.

**Size floors.** Body text belongs at `16px` — that's the browser default, chosen because it's a good reading size, not an arbitrary tradition. Secondary UI text may step down: `14px` for controls and menus (with the mobile-input exception above), `13px` for captions and metadata, and below `12px` lies territory that needs an unusually good excuse.

**Contrast floors (WCAG AA).** Regular-size text must hit `4.5:1` against its background. Large text — approximately `24px` and up, or bold from about `18.5px` — may relax to `3:1`, because bigger and heavier strokes stay legible with less contrast. Check both themes if the product has them; gray-on-gray failures love dark mode.
