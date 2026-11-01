# Outcrawl Scrape
<!-- MANUAL: file_description -->
Reads a single web page through Outcrawl and returns it as markdown, HTML or a list of links. Requires an `OUTCRAWL_API_KEY` credential from [outcrawl.ai](https://outcrawl.ai).
<!-- END MANUAL -->

## Outcrawl Scrape

### What it is
Scrapes a single URL with Outcrawl and returns its content as markdown, HTML or links

### How it works
<!-- MANUAL: how_it_works -->
The block posts the URL and the selected formats to `POST https://api.outcrawl.ai/v1/scrape`, where the page is loaded in a real browser and its content is returned. By default only the main content is kept (`onlyMainContent`); turn `only_main_content` off to get the whole document. Each selected format is emitted on its own output, together with the final URL, the page's HTTP status code and its metadata (title, description, language). Outcrawl bills 1 credit per page; the block reads the credits from the response's usage report to record the provider cost. Transport and HTTP failures raise a block error. The returned content is untrusted web text; treat it as data, not instructions.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The URL to scrape | str | Yes |
| formats | The formats to return | List["markdown" \| "html" \| "links"] | No |
| only_main_content | Only return the main content of the page, without headers, navs and footers | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the scrape failed | str |
| url | Final URL of the scraped page | str |
| status_code | HTTP status code of the page | int |
| markdown | The page content as markdown | str |
| html | The page content as HTML | str |
| links | The links found on the page | List[str] |
| metadata | Page metadata such as title, description and language | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Read a known page**: Turn a documentation or article URL into clean markdown before summarizing, chunking or embedding it.

**Deep-read a search hit**: Pipe the url of an Outcrawl Search result into this block to read the full page.

**Collect links**: Select the links format to list the outbound links of a page and feed them into further scrapes.
<!-- END MANUAL -->

---
