# Performance

Smooth is a property of what you animate and where it runs, not of how
clever the animation is.

## Animate what the compositor owns

`transform`, `opacity`, `filter` and `clip-path` skip layout and paint.
Animating `width`, `height`, `padding`, `margin`, `top` or `left` runs the
whole pipeline every frame. Express movement as a transform and size changes
through one of the height techniques in [motion.md](motion.md#height).

## Name the properties

Transition exactly what changes:

```css
/* Wrong: the browser watches everything, and colors, padding and shadows
   animate when you did not mean them to */
transition: all 150ms ease-out;

/* Right */
transition-property: scale, background-color;
transition-duration: 150ms;
```

Tailwind's bare `transition` is `all`. Use `transition-transform` (which
covers `transform`, `translate`, `scale` and `rotate`), `transition-colors`,
`transition-opacity`, or the bracket form `transition-[scale,opacity,filter]`.

## `will-change`, rarely

It pre-promotes an element to its own compositing layer so the first frame of
a transform does not stutter. Use it only on `transform`, `opacity`, `filter`
or `clip-path`, only after seeing a first-frame hitch, and never as a blanket
on every animated element: each layer costs memory and Safari in particular
will run out. `will-change: all` is never right.

## CSS variables cascade

Changing a custom property on a parent recomputes style for every descendant.
A drag offset written into `--offset` on a list container touches every row
on every pointer move. Write `transform` on the moving element instead.

## Motion libraries under load

A JavaScript animation runs on the main thread. While the browser is parsing
a new route, hydrating, or laying out a big table, it drops frames; a CSS
transition on the same element does not. Two consequences:

- Motion's shorthand `x`, `y` and `scale` values animate on the main thread.
  For an element that must stay smooth while the page is busy, animate the
  `transform` string, or use a CSS transition and let Motion drive only the
  class.
- Anything that plays during navigation (a tab indicator, a page transition,
  a progress bar) is CSS or the Web Animations API, not a rAF loop.

Use JavaScript where it earns its place: springs, gestures, values driven by
the pointer, sequences that depend on measurements.

## The Web Animations API

`element.animate()` gives programmatic control with compositor performance:
interruptible, cancelable, no library:

```js
element.animate([{ clipPath: 'inset(0 0 100% 0)' }, { clipPath: 'inset(0 0 0 0)' }], {
  duration: 600,
  fill: 'forwards',
  easing: 'cubic-bezier(0.77, 0, 0.175, 1)',
})
```

## Layout animations

Motion's `layout` prop is powerful and easy to misuse. It scales the element
to interpolate size, which distorts children (text stretches, radii squash).
Use `layout="position"` when only the position changes, put `layout` on the
children that should not distort, and never wrap a whole page in it.

## Theme switches

A dark mode toggle that transitions every color on the page is a page-wide
repaint for 300ms. Either switch instantly, or transition only the few
surfaces that matter, or use a view transition that crossfades one snapshot.
Never `transition: all` at the root to make the switch feel smooth.

## Blur is expensive

`filter: blur()` and `backdrop-filter` are the costliest things a transition
can do, and Safari pays double. Keep transition blur at 2–4px, never over
12px, and do not animate blur on large surfaces. A glass panel that blurs
what is behind it is fine at rest; animating its blur radius is not.

## Measure before believing

Open the Performance panel, record the interaction, and look for long tasks
and layout thrash during the animation. Then open the Layers panel and count.
If the number of layers surprises you, so will the memory.
