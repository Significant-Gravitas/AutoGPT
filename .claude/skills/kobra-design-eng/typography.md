# Typography

The text is most of the interface. These are the rendering decisions that
make it sit right.

## Wrapping

- **Headings and any short block** (six lines or fewer): `text-wrap: balance`.
  Lines come out even and nothing dangles. Browsers ignore it past their line
  limit, so putting it on body copy does nothing.
- **Paragraphs, captions, descriptions, list items**: `text-wrap: pretty`.
  The last line is never a single orphaned word, and line lengths are
  otherwise left alone. Works at any length.
- **Long-form text, code, preformatted content**: neither. The default is
  right and the layout cost is real.

Tailwind: `text-balance`, `text-pretty`.

## Numerals

Any number that changes while someone is looking at it is tabular:

```tsx
<span className="tabular-nums">{count}</span>
```

Counters, timers, prices, table columns, animated tickers, anything in a
dashboard. Proportional digits make a value jump sideways each time a 1
becomes a 4. Leave static or decorative numbers proportional; a version
string or a phone number gains nothing.

Some fonts redraw the 1 with a wider foot under this setting. That is the
feature working; check it in the project's own face.

## Smoothing

On macOS, text renders heavier than it was drawn. Once, at the root:

```css
html {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
```

Tailwind: `antialiased` on the root element. Never per component, or the
weights disagree across the page. Other platforms ignore it, so it is safe
everywhere.

## Truncation

Decide, per label, whether it wraps or truncates, and make the decision
visible. A truncated label carries its full text in a `title` or a tooltip.
Truncate in the middle for things whose ends matter (file names, hashes,
addresses). Never let a container's width be decided by the longest string
someone might type.

## Text inside controls

- Labels in buttons, tabs and menu items are `whitespace-nowrap`; a button
  that wraps is a broken button.
- A label that changes (a count, a toggle's word) reserves the width of its
  longest value, or the control's neighbors shuffle with every change.
- Line height inside a control is tight (1.2–1.3) so the text centers on the
  control's box rather than on its own leading.
- Weight signals hierarchy, not emphasis. Two weights per surface is usually
  the budget; a third is a reason to look again.

## Measure and rhythm

Body copy sits between 45 and 75 characters per line. Vertical spacing steps
on one scale, and the step between a heading and the block that follows it is
smaller than the step above the heading, so the heading belongs to what comes
after it.

## Captions before code

A line that introduces a block of code or a command ends with a colon. It is
half a sentence and the block is the other half; a full stop closes it before
the thing it is about has appeared.
