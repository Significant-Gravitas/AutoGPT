# Nace Score
<!-- MANUAL: file_description -->
Score text and images on an explicit ordered scale with nace's Drex Max model.
<!-- END MANUAL -->

## Nace Score

### What it is
Score text and up to 8 images with Drex Max using an explicit ordered scale. Levels define the scale: describe concrete evidence for each tier so a strong case does not automatically reach the top. For hiring, distinguish terrible, bad, okay, excellent, brilliant, and one-in-a-thousand by what qualifies.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one `score` question with `levels` as its ordered criteria array, lowest first. Drex Max returns a score on the 0..`max_score` scale (it may be fractional), a probability for each level and a confidence. nace returns the legend as the ordered level list; the block numbers it (`"0"`, `"1"`, ...) so `legend` has the same shape as the Jev Score block's. `max_score` is always the number of levels minus one.

Describe concrete evidence for every level so the scale is calibrated. Evidence goes in `state` and, optionally, in up to 8 PNG, JPEG or WebP `images` referred to as "Image 1", "Image 2" in the question. On failure the block emits an error and no score pins.

On failure, `response` has type `str | None`, and `input_tokens` and `output_tokens` have type `int | None`. Missing values are `None`; no decision output is emitted.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| state | Text or JSON context shared by every question in this stateless call. JSON is compactly serialized; oversized context is truncated with a note. | State | No |
| images | Up to 8 PNG, JPEG or WebP images, each a data URL, an http(s) URL or a workspace file. Refer to them in the question as 'Image 1', 'Image 2' in list order. | List[str (file)] | No |
| question | Plain-language question to score. | str | Yes |
| levels | Ordered level descriptions, lowest first (2-255 levels). Levels define the scale: describe concrete evidence for each tier so a strong case does not automatically reach the top. For hiring, distinguish terrible, bad, okay, excellent, brilliant, and one-in-a-thousand by what qualifies. | List[str] | Yes |

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
| score | Score returned by Drex Max on the 0..max_score scale. | float |
| legend | Level numbers mapped to their descriptions. | Dict[str, str] |
| probabilities | Probability for each score level. | Dict[str, float] |
| confidence | Confidence returned by Drex Max. | float |
| max_score | Highest possible score: number of levels minus one. | int |

### Possible use case
<!-- MANUAL: use_case -->
**Damage assessment**: Grade the severity of damage in a photo on an explicit scale.

**Quality review**: Score how well a screenshot matches a design brief, with each level described by concrete evidence.

**Candidate screening**: Rate written evidence against a calibrated hiring scale.
<!-- END MANUAL -->

---
