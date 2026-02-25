# Firecrawl Map
<!-- MANUAL: file_description -->
Blocks for mapping website structure and discovering all links using Firecrawl.
<!-- END MANUAL -->

## Firecrawl Map Website

### What it is
Firecrawl maps a website to extract all the links.

### How it works
<!-- MANUAL: how_it_works -->
This block uses Firecrawl's mapping API to discover all links on a website without extracting full content. It quickly scans the site structure and returns a comprehensive list of URLs found.

The block is useful for understanding site architecture before performing targeted scraping or for building site maps. Results include both the raw list of links and structured results with titles and descriptions.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The website url to map | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the map failed | str |
| links | List of URLs found on the website | List[str] |
| results | List of search results with url, title, and description | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
**Site Audit**: Map all pages on a website to identify broken links, orphan pages, or SEO issues.

**Crawl Planning**: Discover site structure before deciding which pages to scrape in detail.

**Content Discovery**: Find all blog posts, product pages, or documentation entries on a site.
<!-- END MANUAL -->

---
