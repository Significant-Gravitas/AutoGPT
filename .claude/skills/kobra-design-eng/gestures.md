# Gestures

Anything the person drags, swipes or holds. Direct manipulation is the one
place motion is not optional, because the element has to follow the finger.

## Follow the pointer, then settle

While the pointer is down the element tracks it with no easing at all; any
lag reads as weight in the wrong place. On release, a spring or the drawer
curve carries it the rest of the way, from its current velocity. Springs
here, not durations: a flick and a slow release should not take the same
time.

## Momentum, not just distance

Do not require the drag to cross a threshold. Measure velocity as distance
over elapsed time, and dismiss on either:

```js
const elapsed = performance.now() - dragStart
const velocity = Math.abs(offset) / elapsed
if (Math.abs(offset) >= THRESHOLD || velocity > 0.11) dismiss()
```

A quick flick of twenty pixels is a clearer intent than a slow drag of a
hundred.

## Damping past the edge

Nothing in the physical world stops at a wall. When a drawer is dragged past
its open position, or a list past its end, let it move with increasing
resistance: a fraction of the overshoot, shrinking as the overshoot grows.
It returns on release with the same spring as everything else.

## Pointer mechanics

- **Capture the pointer** when a drag starts (`setPointerCapture`) so the
  gesture continues when the pointer leaves the element's box.
- **Ignore a second touch** once a drag is under way. Without this a second
  finger teleports the element to a new position.
- **`touch-action`** on the draggable so the browser does not scroll the page
  underneath a horizontal swipe (`touch-action: pan-y` for a horizontal
  gesture, `none` for a free one).
- **Dead zone.** A few pixels of movement before a press becomes a drag, so a
  tap with a shaky thumb is still a tap.
- **Keep the transform on the element.** Writing the drag offset into a CSS
  variable on a container recalculates style for every descendant on every
  move. Set `transform` on the thing that moves.

## Spatial consistency

A thing dismisses the way it arrived. A toast that slides in from the bottom
swipes out to the bottom; a sheet that came up from the edge goes back to that
edge. The gesture is the entrance in reverse, so it does not have to be
learned.

## Hold to confirm

Pressing is slow when the person is deciding and fast when the system
answers. A hold-to-delete fills over 1.5–2s, linear, with `scale(0.97)` on
the button, and snaps back in 200ms ease-out the moment the pointer lifts.

## Test on hardware

Simulators lie about touch. Open the dev server on a phone over the local
network and use the remote inspector. Latency, scroll interference and thumb
reach only show up on a real device.
