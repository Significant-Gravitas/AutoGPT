# Treg Catalog
<!-- MANUAL: file_description -->
Search [Treg](https://treg.to)'s catalog of 4,000+ tools across about 120 providers (SEO and backlinks, social, ads, people and company data, scraping, generation) and look up a tool's inputs, price and alternatives before calling it with Treg Call Tool. Both blocks are free. Needs a Treg API key from treg.to (`TREG_API_KEY` for a platform key).
<!-- END MANUAL -->

## Treg Get Tool

### What it is
Look up one Treg tool before calling it: the inputs it takes, its price, and every other provider of the same job. Free.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `GET https://treg.to/catalog/endpoints/<tool id>`. It returns the tool, the query parameters and body fields it takes, and every provider of the same job with its price, success rate and hit rate, so you can choose before you spend. It is free.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| tool_id | The tool's id, e.g. 'treg.people.email.find', as Treg Search Catalog lists it. | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| tool | The tool, its price and reliability. | TregTool |
| inputs | The query parameters and body fields it takes, with notes. | Dict[str, Any] |
| providers | Every provider of the same job, to compare price and reliability before you choose. | List[TregProviderOption] |

### Possible use case
<!-- MANUAL: use_case -->
**Choosing a provider**: Compare price and reliability across providers of one job.

**Building a call**: Read a tool's inputs before wiring Treg Call Tool.
<!-- END MANUAL -->

---

## Treg Search Catalog

### What it is
Search Treg's catalog of 4,000+ tools by the job you need done, with each tool's price and success rate. Searching is free; call a tool you find with Treg Call Tool.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `GET https://treg.to/catalog/search` with your query. Search for the job you need done (for example 'backlinks for a domain'), not a vendor. Each tool comes back with its id, what it does, who serves it, its listed price (empty for routed tools, whose price depends on the provider that answers) and the share of calls Treg saw succeed. Searching is free.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | The job you need done, e.g. 'backlinks for a domain' or 'tiktok comments'. Search for the job, not a vendor. | str | Yes |
| limit | How many tools to return. | int | No |
| added_within_days | Only tools added in the last this-many days. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| tools | Matching tools, best first, with prices and success rates. | List[TregTool] |
| count | How many tools came back. | int |
| total | How many tools matched in all. | int |

### Possible use case
<!-- MANUAL: use_case -->
**Finding a tool**: Let Otto find the right data source for a job, then call it with Treg Call Tool.

**What is new**: List tools added in the last 30 days with `added_within_days`.
<!-- END MANUAL -->

---
