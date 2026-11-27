# MrScraper Scrape HTML
<!-- MANUAL: file_description -->
Fetches the raw HTML of a page, with an optional screenshot. Needs a MrScraper API key: create one at [app.mrscraper.com/api-tokens](https://app.mrscraper.com/api-tokens) and add it as a MrScraper credential.
<!-- END MANUAL -->

## MrScraper Scrape HTML

### What it is
Retrieve the raw HTML of a page. Browser rendering is optional.

### How it works
<!-- MANUAL: how_it_works -->
The block sends `GET https://api.mrscraper.com/` with `html=true`, `markdown=false` and `saveResult=true`, plus the inputs you set; optional inputs you leave empty are not sent. Without `browser_rendering`, MrScraper loads the page without a browser first and switches to a browser if the request is blocked. A `screenshot` other than `false` also sends `browserRendering=true`; the block saves the PNG to your files and outputs it as `screenshot`. The request waits up to `timeout` plus 30 seconds and is sent once, because a repeated scrape can use tokens again; use `max_retries` to let MrScraper retry a blocked page. Page content is untrusted web text; treat it as data, not instructions.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | Absolute HTTP or HTTPS URL of the page to process. | str | Yes |
| browser_rendering | Whether to load the page in a browser: true always does and false never does. Omit it to load the page without a browser first and switch to a browser automatically if the request is blocked. | bool | No |
| timeout | Maximum seconds to wait for the scrape, from 1 to 330. The HTTP request allows 30 additional seconds. | int | No |
| warm_start | Whether to open the site's home page first to warm the session before loading the target URL. | bool | No |
| block_resources | Whether to block images, fonts, and stylesheets while loading the page. | bool | No |
| wait_for_selector | CSS selector or text to wait for on the page before returning the result, up to 500 characters. | str | No |
| action | Scraper action that routes the request or simply flags it, useful for separating analytics per domain. | str | No |
| geo_code | Two-letter country code to load the page from, sent in lowercase. | str | No |
| max_age | Return a cached result of the same request when it is newer than the selected age. A cached result costs the same tokens as the original scrape. | int | No |
| max_retries | Maximum number of retries, as a nonnegative integer. Omit this parameter for no limit. | int | No |
| screenshot | Screenshot to capture with the HTML: false for none, full for the entire page, top for the first viewport, or a WIDTHxHEIGHT viewport from 100 to 3840 pixels with an optional -ZOOM percent from 25 to 300, such as 1280x800 or 1920x1080-75. Setting a screenshot also sends browserRendering=true. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| html | Raw HTML of the page. | str |
| screenshot | PNG screenshot of the page, when a screenshot is set. | str (file) |

### Possible use case
<!-- MANUAL: use_case -->
**Parse a page yourself**: Feed `html` into an HTML or text block to read fields that a fixed parser handles better than an AI model.

**Archive a page with proof**: Set `screenshot` to `full` to keep the HTML and a picture of what the page showed.

**Pages that load content late**: Set `browser_rendering` to true and `wait_for_selector` to the element that holds the data.
<!-- END MANUAL -->

---
