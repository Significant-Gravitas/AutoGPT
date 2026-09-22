# Typesafe Yes No
<!-- MANUAL: file_description -->
Make a Jev yes/no judgment with an optional confidence threshold for an unsure branch.
<!-- END MANUAL -->

## Jev Yes No

### What it is
Ask Jev a plain-language yes/no question and forward data to the chosen pin, or unsure below your confidence threshold.

### How it works
<!-- MANUAL: how_it_works -->
The block asks Jev a Choice question with fixed `yes` and `no` options. It emits the returned confidence and forwards the original `data` on the chosen pin when that confidence meets `min_confidence`. If confidence is below the threshold, only `unsure` receives the data instead. The default threshold is 0.0 and valid thresholds range from 0.0 to 1.0.

Supply the evidence needed for the judgment in the stateless `state` input. The confidence threshold determines the output branch without changing Jev's answer or confidence. Shared outputs expose the exact request, response, usage, timing, and truncation details. Failed calls emit an error without firing `yes`, `no`, or `unsure`.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| state | Text or JSON context shared by every question in this stateless call. JSON is compactly serialized; oversized context is truncated with a note. | State | No |
| question | Plain-language question answerable yes or no. | str | Yes |
| data | Value forwarded unchanged on yes, no, or unsure. | Data | No |
| min_confidence | Emit unsure when Jev's confidence is below this threshold. | float | No |

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
| yes | Data when Jev chooses yes and meets the confidence threshold. | Yes |
| no | Data when Jev chooses no and meets the confidence threshold. | No |
| unsure | Data when confidence is below min_confidence. | Unsure |
| confidence | Confidence returned by Jev. | float |

### Possible use case
<!-- MANUAL: use_case -->
**Human Review Gate**: Forward uncertain judgments to a reviewer while routing confident yes/no decisions separately.

**Request Detection**: Decide whether a support message explicitly requests a particular action.

**Evidence Check**: Test whether a document supplies the evidence for a required criterion.
<!-- END MANUAL -->

---
