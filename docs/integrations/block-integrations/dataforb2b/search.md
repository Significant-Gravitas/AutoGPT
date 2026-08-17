# DataForB2B Search
<!-- MANUAL: file_description -->
Blocks for searching companies and people by structured filters using DataForB2B's B2B database — build target-account and prospect lists for sales, recruiting, and account-based marketing.
<!-- END MANUAL -->

## Company Search

### What it is
Search companies and accounts by structured filters — industry, headcount/size, location, funding, keywords — using DataForB2B's database. Build target-account lists for B2B sales and account-based marketing. Accepts LinkedIn URLs as identifiers.

### How it works
<!-- MANUAL: how_it_works -->
Add as many `filters` entries as you need — each is a column, an operator and a value — and they are validated and combined with `and`/`or` per `match`. Filter values are matched against stored taxonomy values, so resolve them with Search Filter Typeahead rather than guessing: `industry` is `software development`, not `software`. For shapes the list cannot express, such as nested and/or groups, pass a raw `filters_json` (optionally the `applied_filters` output from Smart Search), which is merged with the list via AND. Numeric, boolean, and text columns reject incompatible operators (`=` is accepted on every column, which is why it is the default); `between` requires exactly two comma-separated values. Results are paginated with `count` (clamped to 1-100) and non-negative `offset`. Client and server errors are surfaced via `error`, while a valid search with no matches returns an empty `results` list.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| filters | Filters to apply. Add one per field you want to narrow on; they are combined with 'match' (AND by default). | List[CompanyFilterCondition] | No |
| count | Number of results to return (1-100) | int | No |
| offset | Pagination offset — 0 for page 1, then 25, 50, … to page through results | int | No |
| match | Combine the filters above with 'and' or 'or' | str | No |
| filters_json | Escape hatch for filter shapes the list above cannot express, such as nested and/or groups. Paste 'applied_filters' from Smart Search here with an 'offset' to paginate its results. Merged (AND) with the filters above, or used alone. | Dict[str, Any] | No |
| enrich_live | Fetch fresh live data (uses more credits) | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | Full search response (total, count, results) | Dict[str, Any] |
| results | List of matching companies | List[Any] |
| total | Total number of matches | int |

### Possible use case
<!-- MANUAL: use_case -->
**Target Account Lists**: Build a list of companies matching industry, size, and location criteria for account-based marketing.

**Market Sizing**: Estimate the number of companies matching a given ICP before launching an outbound campaign.

**Funding Research**: Find companies at a particular funding stage or backed by a target investor.
<!-- END MANUAL -->

---

## People Search

### What it is
Search people and B2B leads by structured filters — job title, company, location, industry, seniority, skills — using DataForB2B's database. Find employees at a company, people by job title, who works where, decision-makers and key contacts (owners, founders, C-suite, VPs, directors), and build a prospect or lead list. Accepts LinkedIn URLs as identifiers. The lead-sourcing step of a prospecting or outreach workflow.

### How it works
<!-- MANUAL: how_it_works -->
Add as many `filters` entries as you need — each is a column, an operator and a value — and they are validated and combined with `and`/`or` per `match`. Filter values are matched against stored taxonomy values, so resolve them with Search Filter Typeahead rather than guessing: `industry` is `software development`, not `software`. For shapes the list cannot express, such as nested and/or groups, pass a raw `filters_json` (optionally the `applied_filters` output from Smart Search), which is merged with the list via AND. Numeric, boolean, and text columns reject incompatible operators (`=` is accepted on every column, which is why it is the default); `between` requires exactly two comma-separated values. Results are paginated with `count` (clamped to 1-100) and non-negative `offset`. Client and server errors are surfaced via `error`, while a valid search with no matches returns an empty `results` list.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| filters | Filters to apply. Add one per field you want to narrow on; they are combined with 'match' (AND by default). | List[PeopleFilterCondition] | No |
| count | Number of results to return (1-100) | int | No |
| offset | Pagination offset — 0 for page 1, then 25, 50, … to page through results | int | No |
| match | Combine the filters above with 'and' or 'or' | str | No |
| filters_json | Escape hatch for filter shapes the list above cannot express, such as nested and/or groups. Paste 'applied_filters' from Smart Search here with an 'offset' to paginate its results. Merged (AND) with the filters above, or used alone. | Dict[str, Any] | No |
| enrich_live | Fetch fresh live data (uses more credits) | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | Full search response (total, count, results) | Dict[str, Any] |
| results | List of matching LinkedIn people / leads | List[Any] |
| total | Total number of matches | int |

### Possible use case
<!-- MANUAL: use_case -->
**Prospecting**: Find employees at target companies by job title, seniority, or skill for outbound sales.

**Recruiting**: Search for candidates with a specific title, location, or company background.

**Org Mapping**: Identify decision-makers and key contacts across selected target accounts.
<!-- END MANUAL -->

---
