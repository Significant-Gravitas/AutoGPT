# Data For Seo Related Keywords

### What it is
Get related keywords from DataForSEO Labs Google API.

### What it does
Get related keywords from DataForSEO Labs Google API

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| keyword | Seed keyword to find related keywords for | str | Yes |
| location_code | Location code for targeting (e.g., 2840 for USA) | int | No |
| language_code | Language code (e.g., 'en' for English) | str | No |
| include_seed_keyword | Include the seed keyword in results | bool | No |
| include_serp_info | Include SERP information | bool | No |
| include_clickstream_data | Include clickstream metrics | bool | No |
| limit | Maximum number of results (up to 3000) | int | No |
| depth | Keyword search depth (0-4). Controls the number of returned keywords: 0=1 keyword, 1=~8 keywords, 2=~72 keywords, 3=~584 keywords, 4=~4680 keywords | int | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| related_keywords | List of related keywords with metrics | List[RelatedKeyword] |
| related_keyword | A related keyword with metrics | RelatedKeyword |
| total_count | Total number of related keywords returned | int |
| seed_keyword | The seed keyword used for the query | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Related Keyword Extractor

### What it is
Extract individual fields from a RelatedKeyword object.

### What it does
Extract individual fields from a RelatedKeyword object

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| related_keyword | The related keyword object to extract fields from | RelatedKeyword | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| keyword | The related keyword | str |
| search_volume | Monthly search volume | int |
| competition | Competition level (0-1) | float |
| cpc | Cost per click in USD | float |
| keyword_difficulty | Keyword difficulty score | int |
| serp_info | SERP data for the keyword | Dict[str, True] |
| clickstream_data | Clickstream data metrics | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
