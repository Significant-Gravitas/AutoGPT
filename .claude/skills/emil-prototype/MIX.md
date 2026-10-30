# The Mix Matrix

The picker answers "which direction"; the panel answers "which value"; the matrix answers "which parts". Run it when the user's reaction spans variants — "Scene, but with the trim handles from Trim" — and never make them describe that in prose. A cross-variant merge described in a sentence loses exactly the details that mattered, and the rebuild drifts from what they meant. The matrix makes every part a checkbox with its trade-off next to it.

## When it runs

- `mix <variant>` — explicit.
- The reply to the Phase 6 table names parts of more than one variant. That is a mix, not a riff: run the matrix instead of asking for a description.
- **Not** when the user wants a different direction (`riff`) or a different value (`tune`). Only when they've picked a base and want pieces of the others on it.

## Building the rows

Walk this in order:

1. **The base is fixed.** The variant they named keeps its axis, layout, and every behavior no row replaces.
2. **Diff every other variant against the base.** A candidate is one thing the user can see or do — an element, a behavior, a state — that another variant has and the base lacks or does differently. Implementation differences are not rows.
3. **Keep only portable candidates.** Test: say the base plus this feature in one sentence. If the sentence changes the base's axis — "Dense, but with the Editorial layout" — it isn't portable, it's a different base. Drop it, and if the user asked for it, say the base pick was wrong and offer to mix the other way around.
4. **Merge dependents.** Two features that only make sense together are one row. A feature that needs another one ticked says so in its description: "requires trim handles".
5. **Cap at 8 rows.** Past that the matrix reads as a spec, not a decision, and the user ticks everything. Split into two rounds: mix the top 8 by how much they change the base, build, then mix again.

State what you dropped in step 3 in one line above the matrix, so the user knows the feature was considered rather than forgotten.

## Row content

Four columns, in this order:

| Column | What goes in it | Why |
| --- | --- | --- |
| **Feature** | 2–5 word name, then one line of what the user sees — never how it's built | The name is what gets ticked; the line is what they're agreeing to |
| **Has it** | ✓ or — per non-base variant | The user remembers "the one from Trim", not the feature's name; this column is their index |
| **`<Base>` today** | What the base does instead, in the same terms as the feature line | It turns the checkbox into a trade. Without it, every box gets ticked and the base's character disappears |
| **Choice** | Add · Add as switcher | Below |

Below the rows, one free-text line: "Anything else about how `<Base>` should work?" — it catches the feature you didn't list.

### The switcher option

"Add" replaces the base's behavior. "Add as switcher" keeps both and gives the end user a control to flip between them. A row gets the switcher option only when you can name both modes in two words each — `Scene | Take`, `Trim on | off` — and show those names in the cell. If the modes don't name themselves, it isn't a mode, it's indecision, and offering the switcher lets the user dodge the choice the matrix exists to force.

Every switcher is a mode the product's users have to learn. It is a product decision, not a merge detail: the summary line under the matrix counts them separately ("2 added, 1 of them as a switcher"), and Phase 7 records each one.

## Rendering

Render the matrix with the richest form tool the host has: a widget with real checkboxes and a submit button where the host offers one, a multi-select question tool where it doesn't. If the tool caps options per question, split rows across questions in matrix order and ask about switchers as a second question over the rows they ticked.

The floor is a numbered markdown table in chat with a reply grammar under it: `1, 3s, 4` — row numbers, `s` for switcher. Never go below the floor. Asking "which ones do you want?" in prose puts the user back where they started.

## After the choices

- **Build one merged variant** — the base plus every ticked row plus the free-text line. Nothing else gets in because it was easy to add while you were there; an unticked row is a decision.
- **Name it after the base with a plus** — `Scene+` — and add it to the existing picker as a new pill. The originals stay. A mix that felt right in the matrix can lose on the page, and it can only lose against the pure directions it came from.
- **Switchers are part of the design, not the harness.** They render inside the variant, in the product's materials, where the end user would see them. The control panel from [CONTROLS.md](CONTROLS.md) is for numbers, never for switchers.
- **Re-test the base's worst content** against every ported feature. A feature built inside a variant that happened to sit on short data has never met the base's forty rows.

Present the result as one line per adopted feature: where it landed and what it cost the base, if anything. Then stop — `keep`, `mix`, or `riff` is theirs to call.

## The decision block

After a mix, the Phase 7 block adds a base line, one line per adopted feature with its source and switcher status, and one line per feature left behind with the reason — in the user's words when they gave one:

```md
Scene — decided from /proto/timeline, mixed from Trim and Filmstrip
- Base: Scene (whole-scene bar; shots as segments)
- Adopted from Trim: in/out trim handles, as a switcher (Trim on | off)
- Adopted from Filmstrip + Trim: zoom to this take, as a switcher (Scene | Take)
- Left behind: stop at end of take — running into the next shot is the point of Scene
- Left behind: scene bar at trimmed lengths — no one had trimmed anything yet
```

The left-behind lines are the ones that matter. Without them the same matrix gets rebuilt in three weeks.
