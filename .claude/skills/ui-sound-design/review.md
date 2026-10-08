# Review

How to audit a sound layer, and the shape the report takes.

## Protocol

1. Read the patch and the listener before using the product. List the cues
   and the classification rules.
2. Use the product with sound on at the default level, in order of
   frequency: buttons, menus, toggles, surfaces, sliders, fields, outcomes.
   Note every interaction that sounds and every one that does not.
3. Run the fatigue test: twenty presses of one button, a full slider sweep,
   a menu opened and closed ten times, a six-digit code typed and corrected.
4. Check the silent set: hover, scroll, focus, prose typing, tooltips.
5. Press a disabled control. Press a destructive one. Press a ghost one.
6. Mute, reload, confirm it stayed muted. Unmute, confirm the return sounds.
   Open a second tab and toggle in one.
7. Load the page fresh and press once: confirm it is silent and the second
   press is not.
8. Read the store and the toggle for hydration safety.
9. Record what was actually done. Anything skipped is reported as not
   verified.

## Severity

- `HIGH`: a sound on hover, scroll, focus or prose typing; a cue over 300ms
  on a press; a gain that is tiring within twenty presses; a mute that does
  not persist; a hydration mismatch; sound as the only signal of an outcome.
- `MEDIUM`: a misassigned cue (a toggle playing the generic press, a tab
  playing open); a cue outside the family's register; identical presses with
  no variance; a per-component `play()` call; an interaction that should
  sound and does not.
- `LOW`: envelope or gain polish; naming; a cue that could share more with
  its siblings.

## Output

### Scope

One paragraph: what was in scope, the audio engine, where the patch and the
listener live, the default level used. Then a coverage table:

| Area       | What was inspected                          | Result                                      |
| ---------- | ------------------------------------------- | ------------------------------------------- |
| Vocabulary | cues present, register, family              | n findings, `Clear`, or `Not reviewed: why` |
| Assignment | which interactions got which cue            |                                             |
| Silence    | hover, scroll, focus, prose, tooltips       |                                             |
| Fatigue    | the repeat tests and what they sounded like |                                             |
| Controls   | mute, volume, persistence, tabs, hydration  |                                             |
| Autoplay   | first-press behavior on a fresh load        |                                             |

### Findings

One table, most severe first:

| Severity | Location             | Now                                  | Change                                             | Why                                                         |
| -------- | -------------------- | ------------------------------------ | -------------------------------------------------- | ----------------------------------------------------------- |
| HIGH     | `src/ui/nav.tsx:40`  | `onMouseEnter={() => play('hover')}` | Remove; hover never sounds                         | Fires on every pointer crossing; the first thing turned off |
| MEDIUM   | `src/ui/tabs.tsx:18` | Tab trigger plays `open`             | Classify a trigger with no `aria-expanded` as pick | A tab chooses between things; it does not reveal one        |
| LOW      | `src/sound.ts:71`    | `toggleOff` gain 0.3, same as on     | 0.28                                               | Off is the smaller event; a hair quieter reads as settling  |

### Left alone

| Location          | Candidate                     | Left because                                               |
| ----------------- | ----------------------------- | ---------------------------------------------------------- |
| `src/sound.ts:20` | Round the press attack to 1ms | The edge is the point; rounding it makes a tap into a thud |

### Verified

The interactions performed, at what level, on what output, and what was
heard. Anything not done is **Not verified**.

### Verdict

`Block` while any `HIGH` stands, `Needs changes` for `MEDIUM` or `LOW`,
`Approve` only when nothing actionable remains. List unverified checks
beside it.
