---
name: kobra-design-eng
description: >-
  Design engineering for interfaces that feel finished. Use when building or
  reviewing UI components, deciding whether and how something should animate,
  choosing easing and duration, wiring enter/exit and state transitions,
  tuning hover, press, focus, pending, error and empty states, or fixing
  anything a user describes as "feels off", "janky", "sluggish", "too much",
  or "not quite right". Covers motion, surfaces (radius, shadows, outlines,
  hit areas), typography, icons, gestures, performance and reduced motion.
  Has a build mode and a review mode with quick and full depth.
---

# Kobra Design Eng

Software is mostly good enough now. What separates the tools people reach for
from the ones they tolerate is a long tail of decisions nobody can point to:
the dropdown that arrives a frame sooner, the button that gives a little under
the cursor, the number that does not jitter when it changes. None of these are
noticed on their own. Together they are the difference between an interface
that feels like it is listening and one that feels like it is loading.

This skill is a way of working, not a checklist. Every rule below exists
because it changes what a person feels, and every number in it has been
chosen once so it does not get re-chosen badly under deadline.

## How to work

Four passes, in order. Skipping the first is how interfaces end up animated
everywhere and finished nowhere.

1. **Sense.** Before touching anything, use the interface or read the code the
   way a user meets it. Walk every state: rest, hover, focus, active, pending,
   success, error, disabled, empty. Note where attention snags. Identify the
   project's styling system (Tailwind, CSS modules, CSS-in-JS, a motion
   library) and express every change in that system. Never introduce a second
   one to land a polish fix.
2. **Decide.** For anything that moves, settle three questions before writing
   code: how often the person will see it, what it is for, and what it costs
   them. See [The attention ledger](#the-attention-ledger).
3. **Build.** Use the house numbers below. Deviate only with a reason you would
   be happy to write in a comment.
4. **Prove.** Slow it to 10% in the browser's Animations panel, step frames,
   walk every state again, try it with reduced motion on, on a touch viewport,
   and from the keyboard. Look at it again the next day if you can. What is
   subtly wrong at full speed is obviously wrong at a tenth of it.

## The attention ledger

Motion spends the user's attention. The account is finite and the bill comes on
every repeat, so the more often something happens the less it may cost.

| How often the person meets it                         | What it may do                                                  |
| ----------------------------------------------------- | --------------------------------------------------------------- |
| Every keystroke, shortcut, palette open, list arrow   | Nothing. Instant. No transition on anything keyboard-initiated. |
| Many times an hour: hover, menus, tabs, tooltips      | One property, 120–200ms, ease-out. Feedback, not performance.   |
| A few times a session: dialogs, drawers, toasts, tabs | Standard motion, 200–300ms, with a clear spatial story.         |
| Once: onboarding, first success, a celebration        | May perform. Stagger, spring, a little delight.                 |

Then ask what the motion is _for_. The valid answers are short: it confirms
the interface heard the press, it shows where something came from or went, it
keeps two states from cutting jarringly, it explains a change of state, or it
teaches something once. "It looks nice" is a valid answer only in the last
row of the table.

Motion is never the only channel. Every animated state change also has a
static one: a color, an icon, a label, a position. The person who has reduced
motion on, or who looked away, still knows what happened.

## House numbers

These are decided. Reach for them first.

**Easing.** Stock CSS curves are too shallow to read at UI durations. Use
strong ones, and use a different one for leaving than for arriving:

```css
--ease-out: cubic-bezier(0.23, 1, 0.32, 1); /* arriving, and most things */
--ease-in-out: cubic-bezier(0.77, 0, 0.175, 1); /* moving between two places on screen */
--ease-exit: cubic-bezier(0.4, 0, 0.6, 1); /* leaving */
--ease-drawer: cubic-bezier(0.32, 0.72, 0, 1); /* sheets, drawers, anything dragged */
```

Exits get their own curve for a reason that is easy to miss. A strong ease-out
is front-loaded; that is what makes an arrival land. Point the same curve at
nothing and a 140ms fade is at 10% opacity after three frames, then runs for
eight more with nothing to look at. It reads as a cut. A symmetric curve keeps
the departure on screen for the time it was given. Never use ease-in for
anything the person is waiting on; it delays the first movement, which is the
exact moment they are watching.

**Duration.** Under 300ms for anything interactive. Exits at 60–75% of the
matching enter.

| Thing                        | Duration       |
| ---------------------------- | -------------- |
| Press feedback               | 100–160ms      |
| Tooltip, hover card          | 120–180ms      |
| Menu, select, popover        | 150–220ms      |
| Dialog, sheet, drawer, toast | 200–320ms      |
| Page-level or hero entrance  | up to 500ms    |
| Marketing, explanatory       | whatever reads |

**Press.** `scale(0.97)` on controls 36px and taller, `scale(0.96)` on small
and icon-only ones, never below `0.95`. A CSS transition, 100–160ms ease-out,
so a release mid-press glides back instead of snapping. Skip it on menu
triggers (the menu opening is the feedback) and on anything that already
moves on press, like a push-style button.

**Arrivals.** Never from `scale(0)`. Start at `0.95` or above with `opacity: 0`.
Popovers, menus and tooltips grow from their trigger via `transform-origin`;
dialogs stay centered because nothing on screen owns them.

**Stagger.** 40–80ms between list items, 80–120ms between the chunks of a
hero (title, body, actions). Cap the whole cascade near 600ms. Never gate
input on it.

**Blur.** 2–4px to bridge a crossfade that still reads as two objects.
Never above 12px in a transition; it is expensive, especially in Safari.

**Radius.** Nested corners are concentric: `outer = inner + padding`. Past
about 24px of padding, treat the layers as separate surfaces and pick each
radius on its own. A pill holds pills.

**Hit area.** 44×44 on touch, 40×40 in dense desktop UI, extended with a
pseudo-element when the visible control is smaller. Two hit areas never
overlap.

**Numbers that change** are tabular. **Headings** are `text-wrap: balance`,
body copy `text-wrap: pretty`. **Text** on macOS is `-webkit-font-smoothing:
antialiased` at the root, once.

**Icons** carry the text's weight: 1.5px stroke beside regular text, 2px
beside semibold. One icon set per surface. One SVG per icon, recolored with
`currentColor`; outline by default, filled to mark the active state.

**Pending states** turn the label out of view and a spinner in, through one
shared flip. The label is never rewritten into a status ("Saving…") and never
replaced with a shimmer or marquee. Mark the control `aria-busy` and drop
repeat presses in the handler rather than disabling it, which dims the very
thing that is meant to show the state.

**Errors** shake once, briefly, and say what went wrong in words beside the
field, with `aria-invalid` on the control and `role="alert"` on the message.

## Building a component

Before calling it done:

- Every state exists and was looked at: rest, hover, focus-visible, active,
  pending, success, error, disabled, empty, overflow.
- Nothing animates on first paint. `initial={false}` on `AnimatePresence`
  for things that have a default state; `@starting-style` or a mounted flag
  only for deliberate entrances.
- Transitions name their properties. Never `transition: all`, never
  Tailwind's bare `transition`.
- Interactive state changes use transitions or springs, so they can be
  interrupted. Keyframes are for sequences that run once.
- Hover states are gated behind `@media (hover: hover) and (pointer: fine)`.
- Reduced motion keeps opacity and color, drops travel, scale and 3D, and
  still communicates the state.
- Keyboard works, focus is visible, and the focus ring is the same one the
  rest of the project uses.
- Layout does not shift: numbers are tabular, reserved space is reserved,
  images and embeds have dimensions.
- Dark mode was looked at, not assumed.

## References

| File                             | Read it for                                                                                                |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [motion.md](motion.md)           | Easing and duration in depth, springs, enter/exit, choreography, stagger, height, clip-path                |
| [states.md](states.md)           | Pending, error, success, empty and disabled states; latency thresholds; the static twin every change needs |
| [surfaces.md](surfaces.md)       | Radius, shadows versus borders, image outlines, optical alignment, hit areas, focus rings                  |
| [typography.md](typography.md)   | Wrapping, numerals, smoothing, truncation, text in controls                                                |
| [gestures.md](gestures.md)       | Drag, swipe, momentum, damping, pointer capture, touch specifics                                           |
| [performance.md](performance.md) | Compositor properties, `will-change`, CSS variables, motion libraries under load, WAAPI                    |
| [review.md](review.md)           | The review protocol and the required output format                                                         |

## Reviewing

When asked to review, follow [review.md](review.md). In short: state the
scope and mode, show what was actually walked, report findings in one table
with Severity, Location, Now, Change and Why columns, list what was
considered and left alone, list what was verified and how, and end with a
verdict. Findings are expressed in the project's own styling system, cite a
file and line, and name the user impact rather than the rule.
