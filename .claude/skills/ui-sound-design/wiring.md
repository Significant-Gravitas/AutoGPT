# Wiring

How a cue gets from the patch to the speakers, and how the interface says
which cue it wants.

## One listener, not one per component

Wiring a `play()` into each of sixty components is the same behavior spread
over sixty files, and the sixty-first is silent. Instead: one component
mounted once at the root, holding the audio provider and a single delegated
listener on the document:

```tsx
export function SoundEffects({ children }) {
  const muted = useSoundMuted()
  const volume = useSoundVolume()
  return (
    <SoundProvider enabled={!muted} volume={volume}>
      <SoundEffectListener />
      {children}
    </SoundProvider>
  )
}
```

The listener attaches in the capture phase for `pointerdown`, `keydown`,
`contextmenu`, `input`, and the pointer move and release events a slider drag
needs. `pointerdown` rather than `click`: the sound belongs to the moment the
finger lands, and a click that fires after a handler has already re-rendered
is late.

## Classification from attributes

Components already carry what the listener needs: a `data-slot` naming the
part, `aria-expanded`, `aria-checked`, `aria-pressed`, `data-variant`, roles.
Walk up from the event target to the nearest interactive element and read:

1. **An explicit name.** `data-sound="swoosh"` on a control whose meaning
   only the call site knows. Checked against the patch before use, so a
   typo falls through to the generic press rather than silencing the
   control. Type the attribute's value as the union of cue names so the typo
   is a compile error.
2. **A disabled control.** `:disabled`, `aria-disabled`, `data-disabled`
   answer with `blocked` before anything else is considered.
3. **Toggles by state.** A slot in the toggle list, or anything wearing
   `aria-pressed`, `role="menuitemcheckbox"`, `role="menuitemradio"`, or a
   native checkbox or radio. Read the state before it flips and play the cue
   for where it is going.
4. **Consequence.** `data-variant="destructive"` before the generic branches,
   because a destructive control is also a button.
5. **Suffix rules.** `-close` closes. `-clear` and `-remove` are the flick.
   `-trigger` opens or closes by `aria-expanded`, or picks when there is no
   expanded state (a tab chooses, it does not reveal). `-item`, `-link`,
   `-option` pick, pitched by row. `slider-thumb` and `slider-track` step.
6. **The fallback.** A `button`, `a[href]` or `role="button"` presses, at
   reduced velocity for ghost and link variants.

A label is part of the control it names. Resolve `label.control` before
walking so a click on the text half of a checkbox row sounds the box.

Anything inside a text field, textarea or contenteditable returns nothing.
State it explicitly rather than letting it fall through: the walk climbs, and
a form row wrapping the field would otherwise answer as a pick.

## Keyboard presses

Enter and Space are presses too. Skip repeats, skip text fields except a
palette's search box where Enter runs the highlighted row, and when a listbox
is being driven by `aria-activedescendant`, sound the row it points at rather
than the input that has focus.

## Observed states

Outcomes are renders, not interactions; the press that caused them happened
six digits ago. Nothing dispatches an event when `aria-invalid` appears, so
watch for it:

```ts
const observer = new MutationObserver((records) => {
  for (const record of records) {
    const value = el.getAttribute(record.attributeName)
    const entered = value !== null && value !== 'false' && value !== record.oldValue
    if (record.attributeName === 'aria-invalid' && entered) play('error')
    if (record.attributeName === 'data-success' && entered) play('success')
  }
})
observer.observe(document.body, {
  subtree: true,
  attributes: true,
  attributeOldValue: true,
  attributeFilter: ['aria-valuenow', 'aria-invalid', 'data-success'],
})
```

Only the edge into the state, never the edge out, or clearing a failed field
announces the failure again. Watching attributes rather than components means
every control that reports failure the accessible way is covered without
being listed.

Sliders are the same case. Composite sliders write their value to the
element and dispatch nothing; `aria-valuenow`, which every slider must keep
accurate, is the hook. Filter to `input[type="range"]` and `role="slider"`, or
an indeterminate progress bar chatters on its own.

## The slider ratchet

One tick per value the handle passes over, not one per render. A single
pointer move can carry the value several steps, so count the steps between
the old and new value and sound each one, spaced by the tick's own decay so
no two overlap, and abandoned once the queue falls further behind the handle
than about 60ms. A sweep faster than the spacing saturates rather than
machine-gunning, which is what a real detent does. Pitch rises with the
value so the ends of the range sound like ends.

Separate a drag from a jump: a press on the track that leaps fifty values
passed over nothing and has already sounded once. A few pixels of slop before
a press counts as a drag.

## One path to the speakers

Every `play` goes through one function that adds the per-cue detune wander
and the one-sided velocity variance. It is the only place the randomization
lives, so it cannot be forgotten at a call site, and it is the only place the
cue name is typed, so a renamed cue is a compile error rather than a silent
control.

## The store

Mute and volume live in `localStorage`, read through `useSyncExternalStore`
with a server snapshot of "not muted, default volume." The server cannot
know what a browser stored, and this is what lets the toggle render on the
server without a hydration mismatch. Subscribe to the `storage` event so a
second tab follows.

Keep the two values separate. Folding mute into "volume zero" loses the level
the person had set when they come back.

Guard a hand-edited value: `Number('')` is zero, which is silence with no way
back. Fall back to the default for anything outside `(0, 1]`.

## The toggle

A real button with `aria-pressed` and a label that says which way it goes.
Its own press carries `data-sound` for the last sound on the way out, since
`pointerdown` lands while the provider is still enabled. The return trip
cannot sound at the press, because the provider is still off, so the
listener plays it from an effect on the new value, and skips the initial
render so arriving on the page with sound on stays quiet.

## The autoplay gate

The audio context is suspended until the page has been interacted with.
Resume it on the first gesture and accept that the gesture itself is silent.
Do not pre-warm on load, do not play a "sound enabled" chime, do not show a
banner asking permission. The second press works, and nobody notices the
first.

## Testing without ears

In a headless browser, stub the provider's `play` and assert on the cue
names and options it receives per interaction: a press yields `tap`, a switch
yields `toggleOn` then `toggleOff`, a drag across ten values yields ten
`sliderTick` calls with rising detune, a disabled button yields `blocked`,
a textarea keystroke yields nothing. The vocabulary is testable even though
the sound is not.
