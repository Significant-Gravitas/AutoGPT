# Firecrawl Alexandria
<!-- MANUAL: file_description -->
Blocks for Firecrawl Alexandria, a catalogue of structured-data providers behind the Firecrawl API key: people and company search and enrichment (Apollo, People Data Labs, FullEnrich, Data Legion), jobs, finance and filings, public records, places, retail and thousands more. Finding tools is free; running one costs the Firecrawl credits listed on it.
<!-- END MANUAL -->

## Firecrawl Find Data Tools

### What it is
Finds structured-data tools in Firecrawl Alexandria, free of charge: people and company search and enrichment (work emails, job titles, firmographics via Apollo, People Data Labs, FullEnrich), jobs, finance and filings, public records, places, retail and thousands more. Pass a tool's provider, capability and options to 'Firecrawl Run Data Tool'.

### How it works
<!-- MANUAL: how_it_works -->
This block calls Firecrawl's free `find-tools` capability. With a query it returns the tools that best match the data you describe; with provider or capability IDs it lists those tools; with only category IDs it lists the providers in those categories; with nothing it lists the categories.

Each tool comes back with its `provider` and `capability` IDs, a description, its price in Firecrawl credits and, when contracts are included, the `options` it accepts and the shape of its response. Feed those straight into the Firecrawl Run Data Tool block.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | What data you need, in plain words, e.g. 'work email for a person at a company' or 'sales leaders at Series B software companies'. Leave empty to browse by provider or category. | str | No |
| providers | Limit to these provider IDs, e.g. 'apollo' or 'peopledatalabs'. Alone, lists that provider's tools. | List[str] | No |
| capabilities | Capability IDs to inspect within the providers above, e.g. 'people/search'. Returns each one's full contract. | List[str] | No |
| categories | Category IDs, e.g. 'people' or 'company'. Lists the providers in those categories. | List[str] | No |
| include_contracts | Include each tool's inputs and response shape, so its options can be filled in for 'Firecrawl Run Data Tool'. | bool | No |
| limit | Most results to return (1-100) | int | No |
| offset | Results to skip, for paging | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| results | Matching tools, each with its provider, capability, description, price in Firecrawl credits and, when asked for, its input options. When browsing, providers or categories instead. | List[Dict[str, Any]] |
| result | Each result in turn | Dict[str, Any] |
| level | What the results are: tools, providers, groups or categories | str |
| total | Total number of matches | int |

### Possible use case
<!-- MANUAL: use_case -->
**Lead generation**: Ask for "sales leaders at Series B software companies" and get the people-search tools that can answer it.

**Picking a data source**: Compare the providers and prices available for company enrichment before building a workflow around one.

**Checking tool inputs**: Look up a tool's input contract before adding it to a workflow, so the request fields match what the tool expects.
<!-- END MANUAL -->

---

## Firecrawl Run Data Tool

### What it is
Runs a Firecrawl Alexandria data tool and returns its typed, sourced records: find people by title and company, get verified work emails, enrich a company from its domain, and thousands more. Find the tool and its inputs with 'Firecrawl Find Data Tools' first. Costs the Firecrawl credits listed on the tool.

### How it works
<!-- MANUAL: how_it_works -->
This block sends one Alexandria call (provider, capability and options) to Firecrawl's `/v2/scrape` endpoint and returns the tool's records. The options must match the tool's contract, which Firecrawl Find Data Tools returns.

The call is billed at the Firecrawl credits the response reports, including when the tool fails. Some providers require a Firecrawl organization admin to accept their terms first; the error then includes the dashboard link where that is done.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| provider | Provider ID from 'Firecrawl Find Data Tools', e.g. 'apollo' | str | Yes |
| capability | Capability ID from 'Firecrawl Find Data Tools', e.g. 'people/search' | str | Yes |
| options | The tool's inputs, exactly as its contract declares them (see the tool's options in 'Firecrawl Find Data Tools') | Dict[str, Any] | No |
| timeout | Seconds to wait for the provider (at most 120) | int | No |
| request_id | Only when retrying a call Firecrawl couldn't confirm: the request ID from its error, so the call isn't run and billed twice. Leave empty for a new call. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| data | The records the tool returned | Data |
| records | How many records were returned, as Firecrawl reports it. If it doesn't, the length of a list, 0 for no data, or 1 for any other data | int |
| credits_cost | Firecrawl credits this call cost | float |

### Possible use case
<!-- MANUAL: use_case -->
**Prospect lists**: Find people by title and company, then resolve each to a verified work email.

**Account research**: Enrich a company from its domain with funding, headcount and technologies.

**Compliance checks**: Screen names against sanctions and politically exposed persons lists.
<!-- END MANUAL -->

---
