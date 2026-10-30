# Nace Choice
<!-- MANUAL: file_description -->
Make a typed choice about text and images with nace's Drex Max model.
<!-- END MANUAL -->

## Nace Choice

### What it is
Make a typed choice with Drex Max about text and up to 8 images. Describe what qualifies for each option in plain language.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one `choice` question to nace's System One API with the `options` as its criteria object: each option name maps to a plain-language description of what qualifies. Drex Max returns the winning option, a probability for every option and a confidence, which the block emits as `choice`, `probabilities` and `confidence`.

Put the evidence in `state` and, optionally, up to 8 PNG, JPEG or WebP `images`; refer to them in the question as "Image 1", "Image 2" in list order. The shared outputs expose the exact request and response for auditing. On failure the block emits an error and no decision pins.

On failure, `response` has type `str | None`, and `input_tokens` and `output_tokens` have type `int | None`. Missing values are `None`; no decision output is emitted.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| state | Text or JSON context shared by every question in this stateless call. JSON is compactly serialized; oversized context is truncated with a note. | State | No |
| images | Up to 8 PNG, JPEG or WebP images, each a data URL, an http(s) URL or a workspace file. Refer to them in the question as 'Image 1', 'Image 2' in list order. | List[str (file)] | No |
| question | Plain-language judgment to make. | str | Yes |
| options | Option names mapped to concrete descriptions of what qualifies (2-255 options). | Dict[str, str] | Yes |

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
| choice | Winning option name returned by Drex Max. | str |
| probabilities | Probability for each option. | Dict[str, float] |
| confidence | Confidence returned by Drex Max. | float |

### Possible use case
<!-- MANUAL: use_case -->
**Ticket triage**: Classify a support request, with its attached screenshot, into the team that should handle it.

**Document sorting**: Decide which kind of form a scanned page is before extracting its fields.

**Content moderation**: Choose the policy category an image and its caption fall under.
<!-- END MANUAL -->

---
