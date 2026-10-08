---
name: ui-sound-design
description: >-
  Designing and wiring the sound layer of an interface. Use when adding sound
  to a product, choosing which interactions get a cue and which stay silent,
  designing a family of synthesized cues that sound like one instrument,
  wiring cues through one delegated listener keyed off data attributes,
  building the mute and volume controls, handling the browser's autoplay gate,
  tuning envelopes, pitch, jitter and loudness, or fixing sound that a user
  describes as "annoying", "cheap", "too loud", "samey", "laggy", "plays on
  hover", or "does not play". Has a build mode and a review mode.
---

# UI Sound Design

A good interface sound is one the person does not remember hearing. It
confirms that a press landed, that a switch went the way they meant, that the
thing they submitted was accepted, and then it is gone. Done well, sound
makes an interface feel physical. Done carelessly it is the first setting
anyone turns off, and once they have, nothing you add later will be heard.

This skill is for building a sound layer that earns the right to stay on.
It has opinions about which interactions speak, how the cues relate to each
other, how they reach the speakers, and how loud they are allowed to be.

## How to work

1. **Listen first.** Use the product with sound off and note every
   interaction that changes state. Those are the candidates. Hover, scroll,
   focus and typing prose are not candidates and never will be.
2. **Decide the vocabulary.** Before synthesizing anything, write the list
   of cues as words: press, pick, on, off, open, close, step, confirm, fail,
   warn. Ten to sixteen cues is a full language. More than twenty is a sound
   library nobody can tell apart.
3. **Build one instrument.** Every cue is a variation on one voice, so the
   family reads as a single object being touched in different places. See
   [vocabulary.md](vocabulary.md).
4. **Wire by meaning, not by component.** One listener on the document reads
   what was pressed and picks the cue. Components declare what they are
   through attributes they already carry. See [wiring.md](wiring.md).
5. **Tune at low volume.** Balance at a level where the quietest cue is
   barely there, on laptop speakers, then check on headphones. Anything that
   sounds fine loud is too loud. See [tuning.md](tuning.md).
6. **Run the fatigue test.** Press the same control twenty times in a row.
   Drag a slider end to end. Open and close a menu ten times. If any of that
   is unpleasant, it is the cue's fault, not the person's.

## The rules that hold

**Synthesize, do not sample.** A cue is a few lines of patch: an oscillator,
an envelope, maybe a filter and a second layer. It costs no request, it is
tweakable in one file, it can be detuned per press so no two presses are
byte-identical, and it stays in the same voice as every other cue. A sample
pack is twelve files that never quite match and cannot be varied.

**Short.** A press is 15–30ms of sound. A pick or a toggle is under 100ms. A
surface opening is 100–150ms. Only an outcome (success, failure, warning) is
allowed a second note, and the whole phrase stays under 300ms. One cue in the
set may ring for half a second, and it is reserved for the room changing
under the person, never for anything they pressed.

**Quiet, and quieter when repeated.** Gains between 0.05 and 0.3 of full
scale before the master volume. Cues that fire many times in a gesture (a
slider's detents, a key in a code field) are the quietest things in the set by
a wide margin, short enough that a fast sweep reads as one texture.

**Direction carries state.** On rises, off falls. Open swells up, close
settles down. Stepping into something rises because there is more to come;
committing falls because that was the thing itself. Success climbs to a
resolution, failure drops, warning repeats the same note, which is what
insisting sounds like.

**Weight carries consequence.** A destructive press is lower, slower and
dulled, with a touch of brown noise under it, so it lands instead of clicking.
A ghost button or a link plays the ordinary press at three quarters velocity.
A disabled control still answers, with a dead, pitchless thud, so the press
is acknowledged and refused rather than ignored.

**No two presses match.** Every cue is detuned by a few cents of random
wander per play and its velocity varies by up to ten percent, one-sided, so
a press is only ever softer than designed, never louder. Percussive cues
wander a lot; held intervals barely move, or they go out of tune.

**Never on hover, scroll, focus or prose.** A sound on every pointer crossing
is unlivable. A textarea does not want a soundtrack. The one field that
sounds per keystroke is a one-time code, where each digit is a decision.

**Never the only channel.** Every cue accompanies a visible change. The
person with sound off, or in a library, or on a muted tab, loses nothing but
the texture.

**One press to silence, and it stays silenced.** A mute control in the
chrome, remembered across sessions, shared across tabs. A volume level kept
separately from the mute flag so coming back does not lose where it was set.

**The first press is silent, on purpose.** Browsers hold audio until the page
has been interacted with. Resume the context on the first gesture and accept
that the gesture itself will not sound. Do not fight it with a warm-up trick;
do not show an "enable sound" banner.

## Building a sound layer

Before calling it done:

- A written vocabulary, each cue with the interaction it belongs to.
- One patch file with every cue in it, each one commented with why it sounds
  the way it does.
- One delegated listener; no per-component `play()` calls.
- Classification from attributes the components already carry, with an
  escape hatch attribute for the few controls whose meaning only the call
  site knows, validated against the patch so a typo is a build failure and
  not a silent control.
- Outcomes (invalid, success, value changes) observed from accessibility
  attributes, so any component that reports state the accessible way is
  covered without being enumerated.
- Detune and velocity variance applied in one place, on the only path to the
  speakers.
- Mute and volume in a store the server can render without a hydration
  mismatch.
- A visible toggle with a real label, whose own press is the last sound on
  the way out and whose return is the first sound on the way back.
- The fatigue test passed at laptop volume.

## References

| File                           | Read it for                                                           |
| ------------------------------ | --------------------------------------------------------------------- |
| [vocabulary.md](vocabulary.md) | Choosing the cue set and making it one instrument                     |
| [wiring.md](wiring.md)         | Delegation, attributes, observed states, the store, the autoplay gate |
| [tuning.md](tuning.md)         | Envelopes, oscillators, filters, layers, intervals, jitter, loudness  |
| [review.md](review.md)         | The audit protocol and the required output format                     |

## Reviewing

When asked to review, follow [review.md](review.md): listen through the
product's interactions in order of frequency, report every cue that is
missing, misassigned, too long, too loud, or on an interaction that should be
silent, in one table with Severity, Location, Now, Change and Why columns,
list what was left alone, what was verified and how, and end with a verdict.
