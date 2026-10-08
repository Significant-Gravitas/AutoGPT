# Motion

How things arrive, leave, move and change. Read [SKILL.md](SKILL.md) first for
the attention ledger and the house numbers; this file is the depth behind them.

## Choosing a curve

Decide what the motion is, then the curve follows:

| The element is…                                  | Curve                                              |
| ------------------------------------------------ | -------------------------------------------------- |
| Arriving (menu opens, toast enters, panel grows) | `--ease-out`, `cubic-bezier(0.23, 1, 0.32, 1)`     |
| Leaving (menu closes, toast dismisses)           | `--ease-exit`, `cubic-bezier(0.4, 0, 0.6, 1)`      |
| Moving from one place on screen to another       | `--ease-in-out`, `cubic-bezier(0.77, 0, 0.175, 1)` |
| Being dragged, or settling after a drag          | `--ease-drawer`, `cubic-bezier(0.32, 0.72, 0, 1)`  |
| Changing color or opacity on hover               | `ease`                                             |
| In constant motion (marquee, indeterminate bar)  | `linear`                                           |
| Anything else                                    | `--ease-out`                                       |

Two things people get wrong here:

- **Ease-in on UI.** It starts slow, and the first frames are the ones the
  person is watching. A 300ms ease-in dropdown feels slower than a 300ms
  ease-out one even though both take 300ms. Reserve ease-in for something
  leaving under its own power, and even then prefer the symmetric exit curve.
- **The same curve for exits as for entrances.** A strong ease-out spends
  nearly all its travel in the first third. That lands an arrival. On a
  departure it means the thing is effectively gone after a few frames while the
  transition keeps running, so a fade reads as a cut. The exit curve is
  symmetric: half the travel at half the time, and the whole duration is
  visible.

Do not design curves by hand. Pick from a curve library and adjust one
control point at a time while watching at 10% speed.

## Duration

The interactive ceiling is 300ms. Above it, the interface starts to feel like
it is performing rather than responding. Two corollaries:

- Exits at 60–75% of the matching enter. Attention has already moved on; the
  exit is context, not content.
- Perceived speed is what you are tuning. A spinner that turns faster makes
  the same load feel shorter. A 180ms menu feels more responsive than a 400ms
  one even when the data behind it arrives at the same moment. Once one
  tooltip is open, the next one on the same toolbar opens with no delay and no
  animation.

## Springs

A spring has no duration; it settles. Use one when the motion might be
interrupted, when the value is driven by a pointer, or when the element should
feel like it has mass (a drawer, a card being dragged, a dynamic island).
Springs keep their velocity when retargeted; a CSS keyframe restarts from
zero.

Configure by feel, not by physics:

```js
{ type: 'spring', duration: 0.3, bounce: 0 } // a swap, a flip, a state change
{ type: 'spring', duration: 0.5, bounce: 0.2 } // a drag settling, something playful
```

Bounce above 0.3 is a toy. Zero bounce is right for almost every state change,
because overshoot on a label or an icon reads as a mistake.

Values driven by the pointer are always smoothed through a spring. Tying a
rotation or a parallax directly to `mouseX` looks mechanical because it has no
momentum. `useSpring` in Motion does this in one line. This is decorative
motion, so it belongs on a marketing surface or a hover effect, not on a chart
someone is reading numbers off.

## Arriving and leaving

Nothing real appears from nothing. Start every entrance at `scale(0.95)` or
larger with `opacity: 0`, and give it an origin:

```css
.popover {
  transform-origin: var(--transform-origin); /* Base UI sets this to the trigger side */
  transition:
    opacity 160ms var(--ease-out),
    transform 160ms var(--ease-out);
}
.popover[data-starting-style],
.popover[data-ending-style] {
  opacity: 0;
  transform: scale(0.96);
}
.popover[data-ending-style] {
  transition-timing-function: var(--ease-exit);
  transition-duration: 110ms;
}
```

Dialogs are the exception to origin-awareness: nothing on screen owns them,
so they scale from center.

Use percentages for travel so the numbers survive a redesign. `translateY(100%)`
moves an element by its own height whatever that height is; a drawer hidden at
`translateY(100%)` is hidden at every content size.

For entrances without JavaScript, `@starting-style` replaces the
mounted-flag pattern:

```css
.toast {
  opacity: 1;
  translate: 0 0;
  transition:
    opacity 240ms var(--ease-out),
    translate 240ms var(--ease-out);
  @starting-style {
    opacity: 0;
    translate: 0 16px;
  }
}
```

Keep the `useEffect` mounted flag only where `@starting-style` support is not
enough.

**Exits are softer than entrances.** A small fixed travel (8–16px), a fade, and
the exit curve. Not the full height, not a scale to nothing, not a slide off
screen unless the spatial story needs it (a card returning to the list it came
from, a drawer going back below the fold). And never no exit at all: an element
that vanishes in one frame feels broken even when the person asked for it.

## Choreography

When several things change at once, order them. The eye can follow one
motion; it cannot follow four that start on the same frame.

- **Container first, then contents.** A panel grows to its new height, and
  the content inside fades in a beat later. If content fades while the box is
  still moving, both look wrong.
- **Out before in, unless they overlap on purpose.** With
  `AnimatePresence mode="popLayout"` the leaving element is lifted out of flow
  so the arriving one takes its place without a layout jump. Use `mode="wait"`
  only when the two must not coexist for a single frame.
- **Shared contracts within a family.** Every step of a multi-step dialog
  uses the same shift, the same in and out timing, and the same frame class.
  Two dialogs that step differently are two products.
- **Direction means something.** A step forward slides left; back slides
  right. A toast enters and exits on the same edge so swiping it away feels
  like the reverse of its arrival.

## Stagger

Stagger a group when its appearance is rare enough to afford it and the order
carries meaning (a hero: title, then body, then actions). Do not stagger
routine lists on every render, and never on anything the keyboard opens.
In Motion the parent variant carries the interval and the children only
describe their own two states:

```tsx
<motion.section
  initial="hidden"
  animate="visible"
  variants={{ visible: { transition: { staggerChildren: 0.08 } } }}
>
  {chunks.map((chunk) => (
    <motion.div
      key={chunk.id}
      variants={{
        hidden: { opacity: 0, y: 12, filter: 'blur(4px)' },
        visible: { opacity: 1, y: 0, filter: 'blur(0px)' },
      }}
    />
  ))}
</motion.section>
```

CSS-only, with `animation-delay` stepped per child. Whatever the mechanism,
the last item lands within about 600ms, and the page is interactive from the
first frame.

## Swapping things in place

Two states crossfading in the same spot often look like two overlapping
objects. Three tools, used together:

- **Blur** of 2–4px during the crossing frames. It blends the outgoing and
  incoming so the eye reads one thing transforming.
- **A little scale** on the incoming (0.5 → 1 for an icon, 0.96 → 1 for a
  label) so it has somewhere to come from.
- **A spring with no bounce**, `{ type: 'spring', duration: 0.3, bounce: 0 }`,
  so a swap that reverses mid-flight reverses from where it is.

Icons that change with state (play to pause, copy to check) get this
treatment and stay the same size in the layout. Without a motion library, keep
both icons in the DOM, one absolutely positioned, and cross-fade with CSS using
`cubic-bezier(0.2, 0, 0, 1)`; both directions animate and nothing unmounts.

Labels that become a status are not a swap; see [states.md](states.md).

## Height

Animating `height: auto` is the classic wall. Three ways over it, best first:

1. `interpolate-size: allow-keywords` on the root, then `height` transitions to
   and from `auto` natively. Progressive: browsers without it snap.
2. A grid wrapper going from `grid-template-rows: 0fr` to `1fr`, with the child
   at `min-height: 0` and `overflow: hidden`. Works everywhere, no JS.
3. Measure with a `ResizeObserver` and transition an explicit pixel height on
   the wrapper. The most control, and the way to keep a dialog's frame at the
   height of whichever step is showing.

Whichever you use, the content inside crossfades on its own timing; it does
not stretch with the box.

## clip-path

Clipping is an animation primitive, not a shape tool. `inset()` runs on the
compositor and touches no layout.

- A reveal from the left: `inset(0 100% 0 0)` to `inset(0 0 0 0)`.
- A hold-to-confirm: a colored overlay clipped shut, revealed over 1.5–2s
  linear while pressed, snapped back in 200ms ease-out on release. Slow while
  the person is deciding, fast when the system answers.
- Tabs with a perfect color change: duplicate the tab row, style the copy as
  active, clip it to the active tab's rectangle, and transition the clip. The
  text and background change together because they are one element.
- A before/after slider: two stacked images, the top one clipped to the drag
  position. No extra elements, fully accelerated.

## Transitions versus keyframes

Transitions retarget: change the destination halfway and the element glides
to the new one from wherever it is. Keyframes restart. Anything that can be
triggered again before it finishes (a toggle, a hover, a toast stack, a
tooltip) uses transitions or springs. Keyframes are for sequences that run
exactly once: an entrance, a loading pulse, a success tick.

## Reduced motion

Reduced motion is fewer and gentler, not none. Keep the opacity and color
changes that explain what happened. Remove travel, scale, blur and anything
in 3D. In Tailwind that is `motion-reduce:` on the transition classes; in
Motion, `useReducedMotion()` choosing a flat variant. The state must still be
legible with nothing moving at all.
