# Typesafe Pick Best
<!-- MANUAL: file_description -->
Select an original candidate and rank alternatives using Jev's returned probabilities.
<!-- END MANUAL -->

## Jev Pick Best

### What it is
Choose the best candidate using Jev and rank candidates directly by its probabilities. State the comparison criteria in plain language.

### How it works
<!-- MANUAL: how_it_works -->
The block turns at least two candidates into Choice options named `candidate_1`, `candidate_2`, and so on. Candidate descriptions are compact JSON text capped at 2,000 characters each; any truncation is reported. Jev evaluates those options against the question and state, and the winning label maps back to the unchanged original candidate and its zero-based `best_index`.

`ranked` contains the original candidates ordered directly by descending returned probability, retaining input order for ties. It does not adjust the model's answers. The shared transparency outputs show the exact candidate descriptions that were sent. A failed call emits an error without selecting or ranking candidates.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| state | Text or JSON context shared by every question in this stateless call. JSON is compactly serialized; oversized context is truncated with a note. | State | No |
| candidates | Candidates to compare; descriptions are compact JSON capped at 2,000 characters each. | List[Any] | Yes |
| question | Plain-language comparison, for example: Which candidate best meets the requirements in the state? | str | Yes |

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
| best | Original candidate selected by Jev. | Best |
| best_index | Zero-based index of the selected candidate. | int |
| ranked | Original candidates sorted by returned probability, highest first; ties retain input order. | List[Any] |
| probabilities | Probabilities keyed by candidate_1, candidate_2, and so on. | Dict[str, float] |

### Possible use case
<!-- MANUAL: use_case -->
**Response Selection**: Choose which prepared support reply best addresses the customer's stated problem.

**Search Result Selection**: Rank retrieved passages by how well they support a specific question.

**Action Comparison**: Select among proposed workflow actions using requirements supplied in the state.
<!-- END MANUAL -->

---
