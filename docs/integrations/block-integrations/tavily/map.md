# Tavily Map
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Tavily Map

### What it is
Maps a website's structure with Tavily, discovering its URLs without extracting content

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The root URL to map | str | Yes |
| limit | Maximum number of pages to discover | int | No |
| max_depth | Maximum link depth from the root URL | int | No |
| instructions | Natural language instructions guiding which pages to include (e.g. 'Only documentation pages') | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the map failed | str |
| links | List of URLs discovered on the website | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
