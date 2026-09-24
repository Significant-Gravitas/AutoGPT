# Typesafe Filter
<!-- MANUAL: file_description -->
Filter a bounded list using item-by-item Jev scores and an explicit threshold.
<!-- END MANUAL -->

## Jev Filter

### What it is
Filter items with sequential Jev scores, one call per item. Each call receives JSON context and item and emits its own transparency outputs. Levels define the scale: describe concrete evidence for each tier so a very good case does not reach the top. For hiring, distinguish terrible, bad, okay, excellent, brilliant, and one-in-a-thousand by what qualifies.

### How it works
<!-- MANUAL: how_it_works -->
The block scores items sequentially, making one API call per item with an item-first state containing `item` and `context`. Ordered `levels` define the scale, and a returned score at or above `min_score` places the original item in `passed`; lower scores place it in `rejected`. The final `scores` list stays aligned with the input order. Each attempted item emits its zero-based `item_index` followed by its transparency outputs.

`max_items` defaults to 100 and accepts 1–1,000; larger input lists are rejected before any call. The threshold must fit the configured scale, and level descriptions should define concrete evidence for each tier. Empty input yields three empty lists without calling the API. A failed item stops processing and retains prior transcripts without emitting partial result lists.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| state | Text or JSON context shared by every question in this stateless call. JSON is compactly serialized; oversized context is truncated with a note. | State | No |
| items | Items to score, one sequential API call per item; oversized lists are rejected. | List[Any] | Yes |
| question | Plain-language question to score for each item, using state as context. | str | Yes |
| levels | Ordered level descriptions, lowest first. Levels define the scale: describe concrete evidence for each tier so a very good case does not reach the top. For hiring, distinguish terrible, bad, okay, excellent, brilliant, and one-in-a-thousand by what qualifies. | List[str] | Yes |
| min_score | Keep items with returned score at or above this value. | float | Yes |
| max_items | Maximum number of items allowed in this execution. | int | No |

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
| item_index | Zero-based item index emitted before that item's transparency outputs. | int |
| passed | Items meeting min_score, in original order. | List[Any] |
| rejected | Items below min_score, in original order. | List[Any] |
| scores | Returned scores aligned with the complete input items list. | List[float] |

### Possible use case
<!-- MANUAL: use_case -->
**Evidence Screening**: Keep extracted passages that meet a defined relevance threshold for a research question.

**Ticket Prioritization**: Separate reports that meet a concrete impact rubric from lower-impact reports.

**Catalog Review**: Select product records with sufficiently complete specifications for a downstream workflow.
<!-- END MANUAL -->

---
