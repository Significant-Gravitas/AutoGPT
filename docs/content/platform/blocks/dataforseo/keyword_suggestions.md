# Data For Seo Keyword Suggestions

### What it is
Get keyword suggestions from DataForSEO Labs Google API.

### What it does
Get keyword suggestions from DataForSEO Labs Google API

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Keyword Suggestion Extractor

### What it is
Extract individual fields from a KeywordSuggestion object.

### What it does
Extract individual fields from a KeywordSuggestion object

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| serp_info | data from SERP for each keyword | Dict[str, True] |
| clickstream_data | Clickstream data metrics | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
