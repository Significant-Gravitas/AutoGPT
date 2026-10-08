# Review

How to review an interface with this skill, and the shape the report takes.

## Modes

`full` unless the request says otherwise.

| Mode    | Coverage                                                                                  | Findings reported    | Cap |
| ------- | ----------------------------------------------------------------------------------------- | -------------------- | --- |
| `quick` | The primary path and its highest-traffic states                                           | `HIGH` and `MEDIUM`  | 5   |
| `full`  | The whole requested scope across motion, states, surfaces, typography, icons, performance | All, including `LOW` | 15  |

## Protocol

1. Identify the styling system and any motion library before reading for
   problems. Every finding is written in that system.
2. Walk the states. For each component in scope: rest, hover, focus-visible,
   active, pending, success, error, disabled, empty, overflow.
3. Slow motion to 10% in the Animations panel. Step frames on anything that
   moves. Look for two objects during a crossfade, an origin in the wrong
   place, properties that finish at different times, a curve that starts
   late.
4. Turn on reduced motion. Confirm every state still reads.
5. Use a touch viewport. Confirm hover states do not stick and hit areas are
   reachable.
6. Use the keyboard only. Confirm focus is visible and nothing animates on a
   keystroke.
7. Toggle dark mode.
8. Record what you actually did. Anything not done is reported as not
   verified, never implied.

## Severity

- `HIGH`: makes something inaccessible, misleading, unreadable, or costs the
  person attention on every repeat (a 400ms animation on a keyboard action, a
  focus ring removed, a label that changes width on every tick).
- `MEDIUM`: a noticeable usability or consistency problem (mismatched radii
  on a primary card, `transition: all` on a hot path, exits that cut).
- `LOW`: isolated polish. Reported in `full` mode only.

## Output

### Scope

One short paragraph: mode, what was in scope, framework, styling system,
motion library, and any boundary. Then a coverage table with every category,
even the clean ones:

| Category    | What was inspected                       | Result                                      |
| ----------- | ---------------------------------------- | ------------------------------------------- |
| Motion      | files, components, states, speeds walked | n findings, `Clear`, or `Not reviewed: why` |
| States      |                                          |                                             |
| Surfaces    |                                          |                                             |
| Typography  |                                          |                                             |
| Icons       |                                          |                                             |
| Performance |                                          |                                             |

### Findings

One table, most severe first. Never separate "Before:" and "After:" lines.

| Severity | Location             | Now                                  | Change                                                         | Why                                                                           |
| -------- | -------------------- | ------------------------------------ | -------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| HIGH     | `src/Palette.tsx:41` | 250ms scale-in on ⌘K open            | No transition on open; keep the fade on close only             | Opened dozens of times an hour; every open costs 250ms of waiting             |
| MEDIUM   | `src/Tile.tsx:52`    | `rounded-xl p-3` around `rounded-xl` | Outer `rounded-[20px]`, inner `rounded-lg`                     | Equal nested radii crowd the inner corner                                     |
| MEDIUM   | `src/menu.css:12`    | `transition: all 200ms ease-in`      | `transition: opacity 160ms, transform 160ms` with `--ease-out` | Ease-in delays the first frame; `all` animates colors nobody meant to animate |
| LOW      | `src/Stat.tsx:9`     | `<span>{value}</span>`               | `<span className="tabular-nums">`                              | The value shifts sideways as digits change                                    |

- **Location** cites `path:line`. With no source, cite the screen and the
  component.
- **Now** and **Change** are concrete: what is there and what to write.
- **Why** names the user impact, not the rule number.
- A systemic issue is one row that lists every location.
- Never pad to the cap. Omit what has no findings.

### Left alone

One to three candidates in `quick`, two to five in `full`, that were
considered and rejected, with the reason. Real ones only; if there are fewer,
say so.

| Location          | Candidate            | Left because                                                          |
| ----------------- | -------------------- | --------------------------------------------------------------------- |
| `src/Tile.tsx:52` | Heavier hover shadow | It would break rank with the other tiles that share the surface token |

### Verified

The exact interactions, commands and tools used, and what was observed. Any
step from the protocol that was skipped is listed as **Not verified** with
what remains.

### Verdict

- `Block` while any `HIGH` finding stands.
- `Needs changes` when only `MEDIUM` or `LOW` remain.
- `Approve` only when nothing actionable remains.

List every unverified check beside the verdict. When there are no findings,
skip the findings table, say so plainly, still report what was verified and
what was left alone, and end with `Approve`.
