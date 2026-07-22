# Jina Search
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Extract Website Content

### What it is
This block scrapes the content from the given web URL.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The URL to scrape the content from | str | Yes |
| raw_content | Whether to do a raw scrape of the content or use Jina-ai Reader to scrape the content | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the content cannot be retrieved | str |
| content | The scraped content from the given URL | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Search The Web

### What it is
This block searches the internet for the given search query.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | The search query to search the web for | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| results | The search results including content from top 5 URLs | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
