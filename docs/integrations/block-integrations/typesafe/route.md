# Typesafe Route
<!-- MANUAL: file_description -->
Route unchanged data to one of five fixed output pins using a Jev choice.
<!-- END MANUAL -->

## Jev Route

### What it is
Route data using Jev's typed choice. Plain-language option descriptions define each route; only the winning pin fires.

### How it works
<!-- MANUAL: how_it_works -->
Configure two to five options in the desired order. The first option corresponds to `option_1`, the second to `option_2`, and so on. Jev evaluates the question against the state; only the winning numbered pin emits the original `data`. The block also emits the selected option name, probabilities, and common request transparency.

Option descriptions define the decision criteria in plain language. The numbered pins are fixed even when fewer than five options are configured, and unused pins do not emit data. Calls are stateless and report any truncation of the supplied state. An API failure emits an error without firing a routing pin.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| state | Text or JSON context shared by every question in this stateless call. JSON is compactly serialized; oversized context is truncated with a note. | State | No |
| question | Plain-language judgment deciding the route. | str | Yes |
| options | Two to five option names and qualifying descriptions; dictionary order assigns option_1 to option_5. | Dict[str, str] | Yes |
| data | Value forwarded unchanged on the winning route only. | Data | No |

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
| option_1 | Data when the first configured option wins. | Option 1 |
| option_2 | Data when the second configured option wins. | Option 2 |
| option_3 | Data when the third configured option wins. | Option 3 |
| option_4 | Data when the fourth configured option wins. | Option 4 |
| option_5 | Data when the fifth configured option wins. | Option 5 |
| choice | Winning option name returned by Jev. | str |
| probabilities | Probability for each configured option. | Dict[str, float] |

### Possible use case
<!-- MANUAL: use_case -->
**Team Routing**: Forward a support ticket to the branch corresponding to the responsible team.

**Resume Review Routing**: Send high-rated resume evaluations toward an email review branch and the remaining evaluations toward a sheet.

**Document Processing**: Send a classified document to its configured extraction or review workflow.
<!-- END MANUAL -->

---
