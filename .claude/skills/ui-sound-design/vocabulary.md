# Vocabulary

Which interactions speak, what each one says, and how the set holds together
as one instrument.

## The set

A complete interface language, in the order you will hear it most:

| Cue          | Fires on                                                         | Character                                                             |
| ------------ | ---------------------------------------------------------------- | --------------------------------------------------------------------- |
| press        | Any plain button or link                                         | A 20ms click with a little inharmonic edge. The sound of the product. |
| pick         | Choosing a row: menu item, option, link in a list, tab           | Same shape as press, pitched up; a menu is lighter than a button      |
| on / off     | Switch, checkbox, radio, toggle, checkable menu row              | A short sine sweep: up for on, down for off                           |
| open / close | A surface appearing or leaving: menu, popover, dialog, sheet     | A slower low-passed swell, up to open, down to close                  |
| step         | Grabbing a slider, nudging a value                               | A tiny filtered square tick, near subliminal                          |
| detent       | Every value a slider passes through in a drag                    | Quieter and softer than step; a fast sweep reads as one texture       |
| key          | One digit in a code field                                        | A keypad blip with a bright transient, pitched by position            |
| clear        | Emptying a field, removing a chip                                | The lightest thing in the set: a 30ms rising flick                    |
| commit       | Running a command from a palette; the press that finishes a flow | Falls, with a second voice a fifth above, so it reads as a launch     |
| destructive  | Any control that removes something for good                      | Lower, slower, dulled, with a brown-noise floor                       |
| blocked      | A press on a disabled control                                    | Dead: no pitch movement, no transient, low and absorbed               |
| success      | A form or step accepted                                          | Two notes climbing to a resolution                                    |
| error        | A field or submission rejected                                   | Two notes dropping                                                    |
| warning      | Something that needs attention but did not fail                  | The same note twice, which is what insisting sounds like              |
| copy         | Taking a copy of something                                       | Two identical blips 40ms apart, the way a shutter sounds              |
| notify       | The room changing under the person: theme swap, a new message    | The only cue allowed to ring: two notes a fifth apart, half a second  |

Sixteen is the ceiling. If a new interaction does not map to one of these,
the question is which existing cue it is closest to, not what new sound it
deserves.

## Naming by shape, not by role

Name a cue for what it sounds like when its role might move. "Chirp" and
"swoosh" survive being reassigned; "hover" does not, and a cue named for a
role nobody should use it for is an invitation.

## One instrument

The set reads as a family because every cue shares a lineage:

- **One oscillator vocabulary.** Sine for the cleanest things (press, on and
  off, sweeps), triangle for anything with a little body (pick, open, the
  outcomes), square only for the tick where its edge is the point. Noise is a
  layer under something, never a cue on its own.
- **One register.** Presses live around 1–1.5kHz. Surfaces sit an octave and
  a half lower, 300–650Hz. Outcomes sit between them. Nothing in the set is
  above 2kHz except the tail of a sweep, and nothing is below 170Hz.
- **Three heights of the same figure.** Success, error and warning are one
  instrument at three pitches: the same triangle, the same envelope shape,
  the same 75–90ms gap between notes, separated only by direction. A person
  who has heard one of them recognizes the other two as its siblings.
- **Related cues are related sounds.** Step and detent are the same tick at
  two loudnesses. Clear and commit are a rising flick and a falling one. On
  and off are one sweep in two directions.

## Frequency, then loudness

Rank the cues by how often they will fire in a day. Loudness runs the other
way: the more often, the quieter. A detent at 0.045 gain fires dozens of
times in one drag; a notification at 0.14 fires once an hour. Nothing that
fires on every press sits above 0.2 of full scale before master volume.

## Rows have positions

When a list is picked from, pitch the pick by the row's position, capped so a
long list does not climb into a whistle. The third item is audibly not the
first, and a run of picks becomes a run of related notes. Count raw siblings
even though separators get counted too; being the same every time matters
more than being precise.

## What never sounds

- Pointer entering or leaving anything.
- Scrolling, including reaching the end.
- Focus moving, by keyboard or otherwise.
- Typing into a text field or textarea. A caret landing in a field is not a
  press.
- A tooltip appearing.
- An animation finishing.
- Anything that happens without the person doing something, except the one
  notification cue, used sparingly.
