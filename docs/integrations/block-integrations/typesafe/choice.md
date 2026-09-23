# Typesafe Choice
<!-- MANUAL: file_description -->
Choose among explicitly described options using TypeSafe Jev.
<!-- END MANUAL -->

## Jev Choice

### What it is
Make a typed choice with Jev. Describe what qualifies for each option in plain language.

### How it works
<!-- MANUAL: how_it_works -->
Provide a question, the relevant state, and at least two named options. Each option description explains what qualifies for that answer. Jev selects one configured name and returns its probabilities and confidence; the block forwards those values without changing the model's judgment.

Each call is stateless, so include all needed context in `state`. Structured state becomes compact JSON text, and oversized state is truncated with a visible flag and note. Request and response bodies are exposed alongside timing and token usage. Failed calls emit available transparency and an error without a choice.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| state | Text or JSON context shared by every question in this stateless call. JSON is compactly serialized; oversized context is truncated with a note. | State | No |
| question | Plain-language judgment to make. | str | Yes |
| options | Option names mapped to concrete descriptions of what qualifies. | Dict[str, str] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | API failure details; emitted only when a call fails. | str |
| request | Verbatim JSON request body sent to Jev. | str |
| response | Verbatim UTF-8 response body from Jev, or null if no response arrived. Invalid UTF-8 is preserved as a lossless Base64 data URL with an error. | str |
| latency_ms | Wall-clock API call duration in ms. | float |
| input_tokens | Input tokens reported by the API, or null if unavailable on failure. | int |
| output_tokens | Output tokens reported by the API, or null if unavailable on failure. | int |
| request_id | TypeSafe request ID response header. | str |
| truncated | Whether any input text was truncated. | bool |
| truncation_note | Truncation details, or an empty string. | str |
| choice | Winning option name returned by Jev. | str |
| probabilities | Probability for each option. | Dict[str, float] |
| confidence | Confidence returned by Jev. | float |

### Possible use case
<!-- MANUAL: use_case -->
**Support Classification**: Choose the team responsible for a ticket from clear ownership descriptions.

**Content Tagging**: Assign a document to a controlled set of topics using the supplied definitions.

**Workflow Selection**: Choose which named processing strategy best fits an incoming record.
<!-- END MANUAL -->

---
