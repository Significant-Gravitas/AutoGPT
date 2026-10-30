# Nace Yes No
<!-- MANUAL: file_description -->
Make a Drex Max yes/no judgment about text and images, with an optional confidence threshold for an unsure branch.
<!-- END MANUAL -->

## Nace Yes No

### What it is
Ask Drex Max a plain-language yes/no question about text and up to 8 images, and forward data to the chosen pin, or unsure below your confidence threshold.

### How it works
<!-- MANUAL: how_it_works -->
The block asks Drex Max a `noul` question, whose answer is the probability that the statement is true. It emits that value as `probability`, and `confidence` as the larger of the yes and no probabilities. The original `data` is forwarded on `yes` (probability at least 0.5) or `no` when that confidence meets `min_confidence`; below the threshold only `unsure` receives the data. The default threshold is 0.0 and valid thresholds range from 0.0 to 1.0.

Supply the evidence in the stateless `state` input and, optionally, up to 8 PNG, JPEG or WebP `images` (data URLs, http(s) URLs or workspace files), referred to in the question as "Image 1", "Image 2" in list order. Shared outputs expose the request JSON (image data represented by its MIME type and size), response, usage, timing and truncation details. Failed calls emit an error without firing `yes`, `no` or `unsure`.

On failure, `response` has type `str | None`, and `input_tokens` and `output_tokens` have type `int | None`. Missing values are `None`; no decision output is emitted.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| state | Text or JSON context shared by every question in this stateless call. JSON is compactly serialized; oversized context is truncated with a note. | State | No |
| images | Up to 8 PNG, JPEG or WebP images, each a data URL, an http(s) URL or a workspace file. Refer to them in the question as 'Image 1', 'Image 2' in list order. | List[str (file)] | No |
| question | Plain-language question answerable yes or no. | str | Yes |
| data | Value forwarded unchanged on yes, no, or unsure. | Data | No |
| min_confidence | Emit unsure when Drex Max's confidence is below this threshold. | float | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | API failure details; emitted only when a call fails. | str |
| request | Request JSON sent to nace; inline image data is represented by its MIME type and size. | str |
| response | Verbatim UTF-8 response body from nace, or null if no response arrived. Invalid UTF-8 is preserved as a lossless Base64 data URL with an error. | str |
| latency_ms | Wall-clock API call duration in ms. | float |
| input_tokens | Input tokens reported by the API, or null if unavailable on failure. | int |
| output_tokens | Output tokens reported by the API, or null if unavailable on failure. | int |
| request_id | nace request ID. | str |
| truncated | Whether the state text was truncated. | bool |
| truncation_note | Truncation details, or an empty string. | str |
| yes | Data when Drex Max answers yes and meets the confidence threshold. | Yes |
| no | Data when Drex Max answers no and meets the confidence threshold. | No |
| unsure | Data when confidence is below min_confidence. | Unsure |
| probability | Probability that the answer is yes, from 0 to 1. | float |
| confidence | Confidence in the chosen answer: the larger of the yes and no probabilities. | float |

### Possible use case
<!-- MANUAL: use_case -->
**Visual check**: Decide whether a product photo shows the item described in the listing, and route mismatches to a reviewer.

**Human review gate**: Forward uncertain judgments to a person while routing confident yes/no decisions automatically.

**Evidence check**: Test whether a document or screenshot supplies the evidence for a required criterion.
<!-- END MANUAL -->

---
