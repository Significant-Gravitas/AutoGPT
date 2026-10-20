# Typesafe Score
<!-- MANUAL: file_description -->
Score evidence on a concrete, ordered scale with TypeSafe Jev.
<!-- END MANUAL -->

## Jev Score

### What it is
Score evidence with Jev using an explicit ordered scale. Levels define the scale: describe concrete evidence for each tier so a very good case does not reach the top. For hiring, distinguish terrible, bad, okay, excellent, brilliant, and one-in-a-thousand by what qualifies.

### How it works
<!-- MANUAL: how_it_works -->
Provide a question and at least two level descriptions, ordered from lowest to highest. Jev returns a potentially fractional score, the level legend, probabilities, and confidence. The block emits these values directly and computes only `max_score` as `len(levels) - 1`.

The level text defines what earns each tier. For example, distinguish delivering a tested feature, independently operating production services, and leading measurable improvements across teams; reserve the top tier for evidence beyond ordinary excellent performance. Each call exposes its actual bodies, usage, timing, and truncation status. A failed call emits an error without a score.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| state | Text or JSON context shared by every question in this stateless call. JSON is compactly serialized; oversized context is truncated with a note. | State | No |
| question | Plain-language question to score. | str | Yes |
| levels | Ordered level descriptions, lowest first. Levels define the scale: describe concrete evidence for each tier so a very good case does not reach the top. For hiring, distinguish terrible, bad, okay, excellent, brilliant, and one-in-a-thousand by what qualifies. | List[str] | Yes |

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
| score | Score returned by Jev on the 0..max_score scale. | float |
| legend | Level numbers mapped to their descriptions. | Dict[str, str] |
| probabilities | Probability for each score level. | Dict[str, float] |
| confidence | Confidence returned by Jev. | float |
| max_score | Highest possible score: number of levels minus one. | int |

### Possible use case
<!-- MANUAL: use_case -->
**Role-Fit Review**: Rate documented experience against concrete hiring tiers before a human reviews the result.

**Incident Impact**: Score an incident using levels tied to the number of users or workflows affected.

**Specification Completeness**: Evaluate whether a document provides the evidence required by each completeness tier.
<!-- END MANUAL -->

---
