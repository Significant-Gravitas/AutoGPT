# Dataforseo Keyword Suggestions
<!-- MANUAL: file_description -->
Blocks for getting keyword suggestions with search volume and competition metrics from DataForSEO.
<!-- END MANUAL -->

## Data For Seo Keyword Suggestions

### What it is
Get keyword suggestions from DataForSEO Labs Google API

### How it works
<!-- MANUAL: how_it_works -->
This block calls the DataForSEO Labs Google Keyword Suggestions API to generate keyword ideas based on a seed keyword. It provides search volume, competition metrics, CPC data, and keyword difficulty scores for each suggestion.

Configure location and language targeting to get region-specific results. Optional SERP and clickstream data provide additional insights into search behavior and click patterns.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| keyword | Seed keyword to get suggestions for | str | Yes |
| location_code | Location code for targeting (e.g., 2840 for USA) | int | No |
| language_code | Language code (e.g., 'en' for English) | str | No |
| include_seed_keyword | Include the seed keyword in results | bool | No |
| include_serp_info | Include SERP information | bool | No |
| include_clickstream_data | Include clickstream metrics | bool | No |
| limit | Maximum number of results (up to 3000) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| suggestions | List of keyword suggestions with metrics | List[KeywordSuggestion] |
| suggestion | A single keyword suggestion with metrics | KeywordSuggestion |
| total_count | Total number of suggestions returned | int |
| seed_keyword | The seed keyword used for the query | str |

### Possible use case
<!-- MANUAL: use_case -->
**Content Planning**: Generate blog post and article ideas based on keyword suggestions with high search volume.

**SEO Strategy**: Discover new keyword opportunities to target based on competition and difficulty metrics.

**PPC Campaigns**: Find keywords for advertising campaigns using CPC and competition data.
<!-- END MANUAL -->

---

## Keyword Suggestion Extractor

### What it is
Extract individual fields from a KeywordSuggestion object

### How it works
<!-- MANUAL: how_it_works -->
This block extracts individual fields from a KeywordSuggestion object returned by the Keyword Suggestions block. It decomposes the suggestion into separate outputs for easier use in workflows.

Each field including keyword text, search volume, competition level, CPC, difficulty score, and optional SERP/clickstream data becomes available as individual outputs for downstream processing.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| suggestion | The keyword suggestion object to extract fields from | KeywordSuggestion | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| keyword | The keyword suggestion | str |
| search_volume | Monthly search volume | int |
| competition | Competition level (0-1) | float |
| cpc | Cost per click in USD | float |
| keyword_difficulty | Keyword difficulty score | int |
| serp_info | data from SERP for each keyword | Dict[str, Any] |
| clickstream_data | Clickstream data metrics | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Keyword Filtering**: Extract search volume and difficulty to filter keywords meeting specific thresholds.

**Data Analysis**: Access individual metrics for comparison, sorting, or custom scoring algorithms.

**Report Generation**: Pull specific fields like CPC and competition for SEO or PPC reports.
<!-- END MANUAL -->

---
