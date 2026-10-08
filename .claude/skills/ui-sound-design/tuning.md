# Tuning

The numbers behind the cues, and how to change them without breaking the
family.

## The envelope vocabulary

Every cue is an envelope on an oscillator: attack, decay, sustain, release,
in seconds.

| Cue type      | Attack      | Decay       | Sustain   | Release  |
| ------------- | ----------- | ----------- | --------- | -------- |
| Press, click  | 0–0.001     | 0.015–0.03  | 0         | ≤0.006   |
| Pick, toggle  | 0.001–0.002 | 0.05–0.09   | 0         | 0        |
| Surface swell | 0.004–0.008 | 0.11–0.15   | 0         | 0        |
| Tick, detent  | 0–0.0005    | 0.011–0.014 | 0         | 0        |
| Outcome note  | 0.003–0.004 | 0.13–0.22   | 0         | 0        |
| Ring (notify) | 0.008       | 0.25–0.3    | 0.02–0.03 | 0.1–0.12 |

Attack at zero is a click. A few milliseconds rounds the front off and turns
a click into a tap. Sustain is zero everywhere except the one ringing cue.

## Recipes

**A press with an edge.** A 1.3kHz sine with a frequency-modulation layer at
ratio 0.5 and depth 100, gone in 20ms. The sidebands give it the inharmonic
edge of two things meeting, so it reads as a click without a noise layer
faking one:

```js
tap: {
  source: { type: 'sine', frequency: 1300, fm: { ratio: 0.5, depth: 100 } },
  envelope: { attack: 0, decay: 0.015, sustain: 0, release: 0.005 },
  gain: 0.2,
}
```

**A toggle pair.** One sine sweep, two directions. Up for on, down for off,
the off slightly quieter:

```js
toggleOn:  { source: { type: 'sine', frequency: { start: 520, end: 880 } }, envelope: { attack: 0.002, decay: 0.085 }, gain: 0.3 }
toggleOff: { source: { type: 'sine', frequency: { start: 780, end: 420 } }, envelope: { attack: 0.002, decay: 0.085 }, gain: 0.28 }
```

**A surface.** A triangle sweep through a low-pass around 2.2–2.6kHz, so it
sits behind the sharper cues instead of competing:

```js
open:  { source: { type: 'triangle', frequency: { start: 320, end: 620 } }, filter: { type: 'lowpass', frequency: 2600 }, envelope: { attack: 0.006, decay: 0.13 }, gain: 0.24 }
close: { source: { type: 'triangle', frequency: { start: 560, end: 300 } }, filter: { type: 'lowpass', frequency: 2200 }, envelope: { attack: 0.004, decay: 0.11 }, gain: 0.22 }
```

**Weight.** A low triangle falling from 300 to 170Hz through a 1.4kHz
low-pass, with a brown-noise layer band-passed at 700Hz at a twentieth of
the gain. The noise is the floor the note lands on:

```js
destructive: {
  layers: [
    { source: { type: 'triangle', frequency: { start: 300, end: 170 } }, filter: { type: 'lowpass', frequency: 1400 }, envelope: { attack: 0.002, decay: 0.12 }, gain: 0.32 },
    { source: { type: 'noise', color: 'brown' }, filter: { type: 'bandpass', frequency: 700, resonance: 1.1 }, envelope: { decay: 0.05 }, gain: 0.06 },
  ],
}
```

**A keypad digit.** A short sine blip with a white-noise transient
band-passed high and gone in 10ms, so six in a row read as a sequence being
entered. Pitch each digit by its position (about 45 cents per place) and drop
a backspace well below, so correcting is audibly not entering.

**Outcomes.** Two triangle notes, the second 75–90ms behind the first, on the
same envelope. Success: G5 then D6, climbing a fifth. Error: 300Hz then
224Hz, falling. Warning: 622Hz twice. Pitch the three so they are audibly the
same instrument at three heights.

**A chime that is allowed to ring.** Two triangles a fifth apart, C5 then G5,
the second 120ms behind, both with a long tail. Half a second. Reserved for
the room changing, never for a press.

**A launch.** A triangle falling from C6 to G5 with a sine a fifth above it
arriving 18ms later at a third of the gain, so the pair reads as one sound
with a bloom on it. The falling half of a rising flick: stepping in rises
because there is more to come, committing falls because that was the thing.

## Intervals

- A **fifth up** resolves. Success, chime.
- A **fall** of a fourth or more concludes. Error, commit.
- The **same note twice** insists. Warning.
- A **repeated blip** 40ms apart is a mechanism: shutter, staple, copy.

Stay diatonic within the set. Two cues a tritone apart will sound like two
products.

## Jitter

Per-cue detune wander, in cents, applied on every play:

| Cue                     | Wander | Why                                                                    |
| ----------------------- | ------ | ---------------------------------------------------------------------- |
| Press, blocked          | 26–30  | Percussive; nobody can place the pitch of 20ms                         |
| Pick, key, flick        | 20–24  | Short and bright; takes a lot before it reads as wrong                 |
| Tick, step              | 18     |                                                                        |
| Toggle, commit, detent  | 8–14   | A sweep or an interval; enough to unsample, not enough to sour         |
| Open, close             | 10     | Long enough that wandering pitch reads as an out-of-tune swell         |
| Copy                    | 6      | The 40ms pair is tuned; let it wander and it stops being one sound     |
| Success, error, warning | 5      | A chime that arrives at a new pitch each time stops being recognizable |
| Notify                  | 3      | An actual held interval; it would audibly go flat                      |

Velocity: multiply by `0.9 + random() * 0.1`. One-sided, so a cue is only
ever softer than designed.

## Loudness

- Gains before master volume between 0.045 (detent) and 0.32 (the
  destructive note). Nothing that fires on every press above 0.2.
- Default master volume 0.5. Balance the set at that level on laptop
  speakers in a quiet room, then confirm on headphones that nothing spikes.
- The test that matters: twenty presses in a row at the default level. If it
  is tiring, cut the gain, not the count.

## Changing a cue

Change one parameter at a time and play the whole family after each change,
in the order the person will hear it: press, press, pick, toggle on, toggle
off, open, close. A cue that sounds right alone and wrong in sequence is
wrong. Keep a sound board page in the product that plays every cue on a tile,
so the set can be heard together without walking the app.
