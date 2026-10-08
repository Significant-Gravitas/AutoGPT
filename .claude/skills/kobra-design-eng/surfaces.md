# Surfaces

Corners, edges, depth and the geometry of things people click.

## Concentric corners

Nested rounded elements share a center:

```
outerRadius = innerRadius + padding
```

Equal radii on a parent and its padded child are the single most common thing
that makes a card look crowded at the corners. Calculate every time:

```tsx
<div className="rounded-2xl p-2">   {/* 16px */}
  <div className="rounded-lg">      {/* 8px, because 16 - 8 = 8 */}
```

When the padding passes about 24px the surfaces read as separate objects, and
each radius is chosen on its own. A pill input holds a pill button; a
`rounded-full` group with a `rounded-full` control inside it is concentric by
construction.

## Depth: shadows for lift, borders for structure

A border that exists only to make a card look raised is a border doing a
shadow's job, and it fails on any background it was not drawn against. Use a
layered shadow whose first layer is the hairline:

```css
:root {
  --shadow-ring:
    0 0 0 1px rgb(0 0 0 / 0.06), 0 1px 2px -1px rgb(0 0 0 / 0.06), 0 2px 4px 0 rgb(0 0 0 / 0.04);
  --shadow-ring-hover:
    0 0 0 1px rgb(0 0 0 / 0.08), 0 1px 2px -1px rgb(0 0 0 / 0.08), 0 2px 4px 0 rgb(0 0 0 / 0.06);
}
.dark {
  --shadow-ring: 0 0 0 1px rgb(255 255 255 / 0.08);
  --shadow-ring-hover: 0 0 0 1px rgb(255 255 255 / 0.13);
}
.card {
  box-shadow: var(--shadow-ring);
  transition: box-shadow 150ms ease-out;
}
```

Dark mode collapses to the single ring; the lift layers are invisible on a
dark ground and only add noise.

Keep real borders where they mean something: dividers between rows, table
cell edges, input outlines, and any state (selected, focused, invalid) that a
border communicates. The rule is about purpose, not about the property.

## Image outlines

Photographs and screenshots get a 1px inset outline so their edges read the
same as every other surface, whatever is in the image:

```tsx
<img className="outline -outline-offset-1 outline-black/10 dark:outline-white/10" />
```

Pure black in light mode, pure white in dark, at 10%. Not a tinted neutral from
the palette: a tinted hairline picks up the surface under it and reads as a
smudge along the edge. `outline` rather than `border` so nothing is added to the
box.

## Optical alignment

Geometric center and visual center disagree constantly. When it looks off, it
is off; fix the look.

- **Text with a trailing icon.** The icon side gets 2px less padding than the
  text side, or the icon reads as pushed out. `pl-4 pr-3.5`.
- **A play triangle** sits 1–2px right of center; its mass is on the left.
- **Asymmetric glyphs** (arrows, carets, stars) are best corrected in the SVG
  itself so no component carries a magic margin. Failing that, one `ml-px`.
- **Caps and x-height.** A label beside an icon aligns its x-height, not its
  line box, to the icon's center.

## Hit areas

44×44 wherever a thumb might land, 40×40 in dense desktop interfaces. A
visible 20px control extends itself:

```tsx
<button className="relative size-5 after:absolute after:top-1/2 after:left-1/2 after:size-10 after:-translate-1/2">
```

Two extended hit areas never overlap. When they would, shrink the
pseudo-element to the largest size that does not collide.

## Focus

The focus ring is one design decision made once and used everywhere: the same
color, the same offset, the same width, shown on `:focus-visible` and never on
mouse focus. It is the one border that is not allowed to be subtle. Removing
it to make something look cleaner removes the keyboard from the product.

## Small mechanics

- Controls are `user-select: none` so a double-click does not highlight a
  label.
- Tap highlight is transparent on mobile
  (`-webkit-tap-highlight-color: transparent`) because the press state is the
  highlight.
- `cursor: pointer` on links and on things that behave like links. Buttons in
  an application keep the default cursor unless the project decides otherwise;
  either way, one decision across the surface.
- Scroll containers reserve their gutter (`scrollbar-gutter: stable`) so
  content does not shift when a scrollbar appears.
- Anything with a fixed aspect (video, embed, avatar) declares its dimensions
  so the layout is right before it loads.
