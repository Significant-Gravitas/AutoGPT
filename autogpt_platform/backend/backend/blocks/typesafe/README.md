# TypeSafe Jev blocks

[Jev](https://docs.typesafe.ai/concepts/system-one) is TypeSafe's System One
model for typed judgments: Choice selects an option, Score evaluates a defined
scale, and Noul evaluates a true/false statement. Choice and Score include
probabilities; Noul returns a scalar judgment. Jev does not generate explanations
or other free text. Every call
is stateless: put all relevant evidence and context in `state` each time.
The async SDK uses its default model, `jev-latest`.

Select a **TypeSafe** API-key credential in each block's credentials field.
Keys belong to the user; the provider does not install a shared environment-key
fallback. These blocks charge zero platform credits because they use your own
key. TypeSafe may charge your account: the API reports token usage, not a price
or cost field, and no TypeSafe price is assumed here.

## State and transparency

All seven blocks accept `credentials` and `state: Any = None`. Strings are sent as
text; JSON values become compact JSON text. Jev's roughly 32k-token context is
shared by the **whole request**, including every question and rubric. Since
the SDK does not supply a Jev tokenizer, these blocks conservatively bound
the complete serialized request to 30,976 UTF-8 bytes, reserving 1,024 of the
32,000-byte bound. This is a safety bound, not an exact token count. When
necessary, the state is shortened to a prefix and `truncated` becomes true;
`truncation_note` explains how much was kept. Question/rubric text alone that
cannot fit raises a clear error. Truncated JSON state is a text prefix and
may no longer be a complete JSON document.

Every call emits `request: str` and `response: str | None` as the verbatim
HTTP body strings, plus `latency_ms: float`, `input_tokens: int | None`,
`output_tokens: int | None`, `request_id: str`, `truncated: bool`, and
`truncation_note: str`. On failure, unavailable response/usage values are null
and an `error: str` pin explains the failure; no decision or routing pins fire.
Valid UTF-8 responses remain verbatim. If a response contains invalid UTF-8,
`response` preserves its exact bytes as a `data:application/octet-stream;base64,`
URL, and `error` explains this encoding. Decode the portion after the comma
with Base64 to recover the original bytes; no replacement characters are used.
Timing is measured around the
call, and token counts come from the API. Headers containing credentials are
never outputs. Inspect these pins to see precisely which evidence and rubric
were evaluated.

## Blocks

The following inputs and outputs are in addition to the shared fields above.
The platform also supplies its standard `error` output.

| Block | Inputs | Outputs |
| --- | --- | --- |
| `JevChoiceBlock` | `question: str`, `options: dict[str, str]` | `choice: str`, `probabilities: dict[str, float]`, `confidence: float` |
| `JevScoreBlock` | `question: str`, `levels: list[str]` (lowest first) | `score: float`, `legend: dict[str, str]`, `probabilities: dict[str, float]`, `confidence: float`, `max_score: int` |
| `JevAskManyBlock` | `questions: dict[str, typed question]` (nonempty) | `answers: dict[str, dict[str, Any]]`, one typed answer per key |
| `JevRouteBlock` | `question: str`, `options: dict[str, str]` (2–5), `data: Any = None` | Winning `option_1`…`option_5: Any` forwards `data`; `choice: str`, `probabilities: dict[str, float]` |
| `JevYesNoBlock` | `question: str`, `data: Any = None`, `min_confidence: float = 0.0` (0–1) | `yes: Any` or `no: Any` forwards `data`; `unsure: Any` forwards it instead below the threshold; `confidence: float` |
| `JevPickBestBlock` | `candidates: list[Any]`, `question: str` | `best: Any`, `best_index: int` (zero-based), `ranked: list[Any]`, `probabilities: dict[str, float]` |
| `JevFilterBlock` | `items: list[Any]`, `question: str`, `levels: list[str]`, `min_score: float`, `max_items: int = 100` | `passed: list[Any]`, `rejected: list[Any]`, `scores: list[float]` aligned with items; per-call `item_index: int` |

Route pin numbering follows option insertion order. PickBest uses labels
`candidate_1`, `candidate_2`, etc.; its ranking comes directly from returned
probabilities, returning the original candidates in descending probability
order (ties keep input order). Each candidate description is compact JSON
limited to 2,000 characters, with any truncation reported on the common pins.
Choice/Route require at least two options, Score/Filter at least two levels,
and PickBest at least two candidates. Questions must be nonempty strings.
Filter accepts `max_items` from 1 to
1,000, and `min_score` from 0 to `len(levels) - 1`.

Filter evaluates `{"item": item, "context": state}` sequentially,
one request per item, and rejects inputs above `max_items` before making any
calls. For each item it emits `item_index` followed by the transparency pins;
connect them together when recording a per-item audit. Its final result lists
preserve input order. An empty item list yields empty result lists and makes
no API call, so there are no per-call transparency outputs in that case.
If an item call fails, earlier transcripts remain available, but Filter emits
no partial result lists that could be mistaken for a complete evaluation.

AskMany accepts these question shapes and sends them together in one request:

```json
{
  "team": {"type": "choice", "question": "Which team owns this?", "options": {"billing": "Payment problems", "support": "Other product help"}},
  "urgency": {"type": "score", "question": "How urgent is this?", "levels": ["No blocked work", "One person blocked", "An entire team blocked"]},
  "refund": {"type": "noul", "question": "Is a refund requested?"}
}
```

Noul also accepts `criteria: {"true": "...", "false": "..."}`. Its answer is
`{"type": "noul", "noul": float}`, without probabilities or confidence.
Choice option descriptions
define each answer; Score level descriptions define what earns each tier.

## Calibrate the scale with evidence

For hiring, avoid a bare scale such as "bad / okay / good." Describe concrete
evidence at every level so that a very good resume does not automatically earn
the maximum. For example: **0 terrible** — no demonstrated programming;
**1 bad** — introductory exercises only; **2 okay** — delivered a tested
backend feature under guidance; **3 excellent** — independently owns production
Python services; **4 brilliant** — leads complex systems with measurable
improvements across teams; **5 one-in-a-thousand** — sustained, externally
verifiable, field-leading contributions with exceptional impact. A Score can
be fractional; `max_score` is always `len(levels) - 1`.

## Example agent and optional live check

Import [`resume-review.example.json`](resume-review.example.json), which uses
the repository's agent graph format:

```text
Resume file → FileRead (text) → JevAskMany → JevRoute
                                            ├─ option_1 → format → reviewer email
                                            └─ option_2 → format → row → Google Sheet
Results spreadsheet (Google picker) ────────────────────────────────┘
```

Use a UTF-8 `.txt` resume; PDF/DOCX need an appropriate extraction block first.
The example has no saved credentials. Choose TypeSafe credentials on both Jev
nodes, configure SMTP and replace the deliberate `.invalid` recipient/server,
and choose a spreadsheet through the Google picker. The platform must have
Google OAuth configured and agent input subtype blocks enabled (the default).
After setup, running the graph sends an email or appends a row. AskMany's
typed ratings are both Route's state and its forwarded data; the two formatting
steps only adapt that data for the destination blocks.

All automated tests mock the API. To explicitly make seven real requests,
run from `autogpt_platform/backend`:

```powershell
poetry run python -m backend.blocks.typesafe._integration --env-file "C:\path\to\.env" --output "jev-integration.json"
```

Alternatively, set `TYPESAFE_API_KEY` and omit `--env-file`. The script runs
each block once with synthetic data, uses one item for Filter, prints every
output pin (including verbatim bodies), and optionally saves them as JSON.
It stops on a failed block, preserving completed blocks in the optional report.
It reads the key internally, never prints it or rewrites the dotenv file, and
does nothing on import. It does not execute the example email/sheet graph.
