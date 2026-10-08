# States

An interface is mostly states, and most of them are invisible until something
goes wrong. Build every one on purpose.

## The full set

Every interactive component has these, and each was looked at before it
shipped:

| State         | What it needs                                                                                            |
| ------------- | -------------------------------------------------------------------------------------------------------- |
| Rest          | The default. Nothing animates into it on first paint.                                                    |
| Hover         | One property changing, ≤150ms, and only on devices that hover.                                           |
| Focus-visible | The project's focus ring, on the keyboard only, never suppressed.                                        |
| Active        | Press feedback: the scale, or the travel for a push button. 100–160ms.                                   |
| Pending       | The label turns out and a spinner turns in. `aria-busy`. Repeat presses dropped.                         |
| Success       | A static cue (icon, color, copy) plus, at most, one small motion. Reverts on its own if it is transient. |
| Error         | A single shake, the field marked `aria-invalid`, and words that say what to do next.                     |
| Disabled      | Visibly inert, `aria-disabled` or `disabled`, and a reason nearby if the reason is not obvious.          |
| Empty         | A sentence that says what will be here and the action that puts it there. Not a blank panel.             |
| Overflow      | Long labels truncate or wrap by decision, and the layout holds. Scroll regions show that they scroll.    |

## Pending

The one that most often ships wrong. Rules:

- **The label is not rewritten.** "Send invite" does not become "Sending…".
  The word on the button is the thing the person chose, and taking it away at
  the moment they commit to it hands them a different sentence to read.
  Three dots are a stand-in for motion where real motion costs nothing.
- **No shimmer, marquee or sweep** across the control. Progress that is not
  measured is not shown as a bar.
- **One shared flip.** The label pitches out of view and a spinner pitches
  in, on a spring with no bounce, through one component every busy button in
  the project uses. Two buttons that flip differently are two products.
- **`aria-busy="true"`** on the control, and repeat presses ignored in the
  handler. Not `disabled`, which dims the spinner that is meant to be the
  state.
- **Reduced motion** trades the flip for a plain crossfade; the spinner still
  turns.

For content rather than actions, a skeleton in the shape of what is coming.
Show it only after ~150ms (a load that finishes sooner never needed one) and
keep it up at least 300ms once shown, so a fast response does not flash.

## Latency thresholds

| The response arrives within | What the person needs                                                 |
| --------------------------- | --------------------------------------------------------------------- |
| 100ms                       | Nothing. It feels instant.                                            |
| 300ms                       | Press feedback is enough; do not show a spinner that will only flash. |
| 1s                          | A pending state on the control that was pressed.                      |
| 10s                         | Progress, or an honest estimate, and a way to leave.                  |

Optimistic updates move the first row down a level: show the result, mark it
provisional, reconcile when the server answers, and animate the correction if
there is one rather than snapping.

## Error

The shake is one gesture, about 400ms, three or four small horizontal
excursions with a strong ease-out, on the field or the dialog that owns it. It
runs once per submission and never loops. Alongside it:

- `aria-invalid="true"` on the control, which also drives the ring color.
- A message with `role="alert"` next to the field, in words: what was wrong
  and what to do. "Invalid" is not a message.
- The error clears the moment the input changes, not on the next submit.
- Focus moves to the first invalid field.

A destructive confirmation that asks the person to retype something treats a
mismatch the same way: shake, mark, say why.

## Success

Success is mostly static. The icon changes, the color changes, the copy
changes. Add motion only where a single small movement makes the change
legible (a check drawing itself, a like popping once) and let it revert
quietly if the state is temporary, on a timer the person can read.

A copy button that becomes a check for two seconds is the model: the swap is
an in-place icon transition (see [motion.md](motion.md#swapping-things-in-place)),
the label does not change width, and it comes back on its own.

## Empty

Empty states are the first thing a new user sees, so they are the one screen
that may explain itself. One sentence about what will be here, one action that
starts it, and the same layout the populated state will have so the transition
from empty to first item is not a redesign.

## Disabled

A disabled control that gives no reason is a dead end. If the reason is not
obvious from context, put it in a tooltip or a line of helper text. Disabled
controls do not shrink on press, do not change on hover, and are dimmed
through opacity so their text and icon dim together.

Prefer `aria-disabled` with the handler dropping the action when the control
should still be focusable and explain itself; use `disabled` when it should
leave the tab order.

## Hover, on the right devices

Touch devices fire hover on tap and hold it there. Gate every hover style:

```css
@media (hover: hover) and (pointer: fine) {
  .item:hover {
    background: var(--muted);
  }
}
```

In Tailwind 4 the `hover:` variant already carries this media query. In
Tailwind 3 it does not; configure `future.hoverOnlyWhenSupported` or write the
query.

## Every change has a static twin

Motion is a channel, never the only one. Whatever animates also changes
something that holds still: a color, an icon, a label, a badge, a position.
Someone who looked away for the 200ms, or who runs with reduced motion, or
who is using a screen reader, still learns what happened.
