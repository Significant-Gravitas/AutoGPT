# Typesafe Ask Many
<!-- MANUAL: file_description -->
Ask several typed questions about one shared state with TypeSafe Jev.
<!-- END MANUAL -->

## Jev Ask Many

### What it is
Ask multiple Choice, Score, or Noul questions of one shared state with Jev in a single call. Levels define the scale: describe concrete evidence for each tier so a very good case does not reach the top. For hiring, distinguish terrible, bad, okay, excellent, brilliant, and one-in-a-thousand by what qualifies.

### How it works
<!-- MANUAL: how_it_works -->
The block submits every entry in `questions` in one API request with the same state. Each named entry selects Choice (`question` and `options`), Score (`question` and ordered `levels`), or Noul (`question` and optional `true`/`false` criteria). The `answers` output preserves the question keys and returns each typed answer as a plain dictionary. Noul returns a scalar value without invented confidence or probability fields.

At least one question is required. Describe concrete evidence for every Score tier, reserving the highest tier for exceptional evidence. Questions share the request budget; oversized state is truncated with an explicit note. The transparency outputs record the actual request and response, and a failed call emits an error instead of answers.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| state | Text or JSON context shared by every question in this stateless call. JSON is compactly serialized; oversized context is truncated with a note. | State | No |
| questions | Named typed judgments sharing one state and one API call: choice uses question/options, score uses question/levels, noul uses question and optional true/false criteria. Levels define the scale: describe concrete evidence for each tier so a very good case does not reach the top. For hiring, distinguish terrible, bad, okay, excellent, brilliant, and one-in-a-thousand by what qualifies. | Dict[str, Any] | Yes |

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
| answers | Typed answers as plain dictionaries, keyed by question name. | Dict[str, Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
**Resume Review**: Evaluate production experience and a calibrated role-fit score together before routing the ratings to a reviewer.

**Support Triage**: Classify the owning team, score urgency, and check whether a refund was requested in one call.

**Document Screening**: Ask several independent evidence questions about the same extracted document text.
<!-- END MANUAL -->

---
