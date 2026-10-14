# OpenUI journeys

Frozen user journeys for checking when Copilot renders an OpenUI view and whether the view is correct. Each line of a `.jsonl` file is one case: an `id`, `title`, `kind` (`positive`, `negative` or `boundary`), `split` (`development` or `final-check`) and one or more `turns`. Each turn has a `prompt`, an optional `ui` expectation (`required`, `forbidden` or `optional`) and the `criteria` a reply must meet.

| File | Cases | Turns |
| --- | --- | --- |
| `journeys.jsonl` | 24 | 39 |
| `validation-journeys.jsonl` | 17 | 28 |
| `connected-journeys.jsonl` | 12 | 12 |
| `reliability-journeys.jsonl` | 29 | 44 |
| `correctness-journeys.jsonl` | 40 | 58 |
| `capability-journeys.jsonl` | 25 | 30 |

`correctness-journeys.jsonl` contains the reliability cases plus newer ones.

To run a case, send each turn in order to a fresh, signed-in Copilot session and judge the replies and views against the criteria. Keep three questions apart: did the model make the right choice to show a view or not, does the view parse, and is the task answered correctly. Built-in clarifying questions are fine when information is missing.

These cases do not call a model during the unit test suite. To check recorded outputs, run these from `autogpt_platform/frontend`:

- `scripts/evaluate-openui.ts <dir> [run]` grades a recorded run (`<dir>/cases.jsonl` plus `<dir>/runs/<run>/cases/<id>.json`) for completion, the view choice, parser validity and expected components. Journey turns need converting to that layout first.
- `scripts/inspect-openui-journeys.ts` validates recorded sources and exports their parsed props for review.
