# DataForB2B Typeahead
<!-- MANUAL: file_description -->
Resolve free-text queries into the exact filter values DataForB2B's search endpoints expect, so structured search filters match real records instead of failing silently on typos or near-misses.
<!-- END MANUAL -->

## Search Filter Typeahead

### What it is
Resolve the exact filter value (company, industry, job title, skill, school, investor, location, category) for people and company searches with DataForB2B.

### How it works
<!-- MANUAL: how_it_works -->
The block trims and validates the free-text query `q`, then sends it with `filter_type` to DataForB2B's typeahead endpoint. Use the returned `values` as a People Search or Company Search filter rather than guessing at exact spellings — the search endpoints match on stored taxonomy values, so a plausible-looking guess silently returns zero results (`industry` is `software development`, not `software`). Industry is split per-side upstream: resolve `people_industry` for People Search and `company_industry` for Company Search, since the two taxonomies differ in both membership and casing. `limit` is clamped to 1-20, client and server errors are surfaced via `error`, and a valid query with no match returns empty `results` and `values` lists.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| filter_type | Filter type to resolve. Industry is split per-side: use people_industry for People Search and company_industry for Company Search | "company" \| "people_industry" \| "company_industry" \| "title" \| "skill" \| "school" \| "investor" \| "location" \| "city" \| "region" \| "category" | Yes |
| q | Free-text query to resolve | str | Yes |
| limit | Max suggestions (1-20) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | Full typeahead response | Dict[str, Any] |
| results | List of suggestions | List[Dict[str, Any]] |
| values | Resolved stored values | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Filter Autocomplete**: Resolve a user-typed company or industry name to the exact stored value before running a structured People/Company Search.

**Search Validation**: Confirm a job title or location exists in DataForB2B's taxonomy before building a filter around it.

**Query Normalization**: Convert user-entered company, school, or skill names into canonical values for repeatable searches.
<!-- END MANUAL -->

---
