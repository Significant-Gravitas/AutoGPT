# nace Drex Max blocks

[Drex Max](https://console.nace.ai/docs/drex/guides/drex-max) is nace's decision
model. It answers typed questions about text and images through the
[System One](https://console.nace.ai/docs) API: Choice selects an option, Score
evaluates a defined scale, and Noul judges a yes/no statement. Choice and Score
include probabilities and a confidence; Noul returns a probability. Drex Max
does not generate explanations or other free text. Every call is stateless: put
all relevant evidence in `state` (and `images`) each time.

Select a **nace** API-key credential in each block's credentials field. Keys
belong to the user; the provider installs no shared environment-key fallback.
These blocks charge zero platform credits because they use your own key. nace
bills your account per input token (the API reports token usage, not a price).

The blocks mirror the TypeSafe Jev blocks' inputs and outputs, so a graph can
swap one for the other. The differences are the `images` input, the yes/no
block's extra `probability` output, and the model: calls go to `drex-max`
(`drex-latest` is a text-only sibling that rejects images).

## State, images and transparency

All three blocks accept `credentials`, `state: Any = None` and
`images: list[MediaFileType] = []`. Strings are sent as text; JSON values become
compact JSON text. The state is bounded to 131,072 UTF-8 bytes, a conservative
limit well under the model's token limits (a token never covers less than a
byte; drex-max allows 262,144 input tokens per question), and a longer state is cut to a prefix with `truncated` set
and `truncation_note` explaining how much was kept. Truncated JSON state is a
text prefix and may no longer be complete JSON.

`images` take up to 8 PNG, JPEG or WebP images, each a data URL, an http(s) URL
or a workspace file; the platform converts them to data URLs before the call.
Refer to them in the question as "Image 1", "Image 2" in list order. The whole
request body is limited to 4 MiB; a larger one is rejected before any call.

Every call emits `request: str`, the request JSON with inline image data represented
by its MIME type and size. The verbatim response body is `response: str | None`.
Other shared outputs are `latency_ms: float`, `input_tokens: int | None`,
`output_tokens: int | None`, `request_id: str`, `truncated: bool` and
`truncation_note: str`. On failure, unavailable response/usage values are null
and an `error: str` pin explains the failure; no decision or routing pins fire.
A 529 overloaded response is reported, not retried. If a response contains
invalid UTF-8, `response` preserves its exact bytes as a
`data:application/octet-stream;base64,` URL and `error` says so. Headers
containing credentials are never outputs.

## Blocks

The following inputs and outputs are in addition to the shared fields above.
The platform also supplies its standard `error` output.

| Block | Inputs | Outputs |
| --- | --- | --- |
| `NaceChoiceBlock` | `question: str`, `options: dict[str, str]` (2–255) | `choice: str`, `probabilities: dict[str, float]`, `confidence: float` |
| `NaceScoreBlock` | `question: str`, `levels: list[str]` (2–255, lowest first) | `score: float`, `legend: dict[str, str]`, `probabilities: dict[str, float]`, `confidence: float`, `max_score: int` |
| `NaceYesNoBlock` | `question: str`, `data: Any = None`, `min_confidence: float = 0.0` (0–1) | `yes: Any` or `no: Any` forwards `data`; `unsure: Any` forwards it instead below the threshold; `probability: float`, `confidence: float` |

Choice sends `options` as the `criteria` object of a `choice` question and
returns the winning option with its probabilities. Score sends `levels` as the
`criteria` array of a `score` question; nace returns the legend as that ordered
list, which the block numbers (`"0"`, `"1"`, …) to match Jev's `legend`. A score
can be fractional; `max_score` is always `len(levels) - 1`.

Yes/No asks a `noul` question, whose answer is the probability that the
statement is true. `probability` is that value; `confidence` is the larger of
the yes and no probabilities, and `unsure` fires instead of `yes`/`no` when it
is below `min_confidence`. A probability of exactly 0.5 counts as yes.

## Calibrate the scale with evidence

For hiring, avoid a bare scale such as "bad / okay / good." Describe concrete
evidence at every level so that a strong resume does not automatically earn
the maximum: **0 terrible** — no demonstrated programming; **1 bad** —
introductory exercises only; **2 okay** — delivered a tested backend feature
under guidance; **3 excellent** — independently owns production Python
services; **4 brilliant** — leads complex systems with measurable improvements
across teams; **5 one-in-a-thousand** — sustained, externally verifiable,
field-leading contributions.

## Tests

All automated tests mock the HTTP boundary; none call nace. From
`autogpt_platform/backend`:

```text
poetry run pytest backend/blocks/nace
```
