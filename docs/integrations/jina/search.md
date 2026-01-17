# Jina Search
<!-- MANUAL: file_description -->
Blocks for extracting website content and performing web searches using Jina AI.
<!-- END MANUAL -->

## Extract Website Content

### What it is
This block scrapes the content from the given web URL.

### How it works
<!-- MANUAL: how_it_works -->
The block sends a request to the given URL, downloads the HTML content, and uses content extraction algorithms to identify and extract the main text content of the page.
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
A data analyst could use this block to automatically extract article content from news websites for sentiment analysis or topic modeling.
<!-- END MANUAL -->

---

## Search The Web

### What it is
This block searches the internet for the given search query.

### How it works
<!-- MANUAL: how_it_works -->
The block sends the search query to a search engine API, processes the results, and returns them in a structured format.
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
A content creator could use this block to research trending topics in their field, gathering ideas for new articles or videos.
<!-- END MANUAL -->

---
