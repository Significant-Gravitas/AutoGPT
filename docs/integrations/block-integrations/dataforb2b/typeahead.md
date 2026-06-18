# Dataforb2B Typeahead
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Search Filter Typeahead

### What it is
Resolve the exact filter value (company, industry, job title, skill, school, location) for people and company searches with DataForB2B.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| type | Filter type to resolve (company, industry, title, skill, school, investor, location, category) | str | Yes |
| q | Free-text query to resolve | str | Yes |
| limit | Max suggestions (1-20) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the lookup failed | str |
| result | Full typeahead response | Dict[str, Any] |
| results | List of suggestions | List[Any] |
| values | Resolved stored values | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
