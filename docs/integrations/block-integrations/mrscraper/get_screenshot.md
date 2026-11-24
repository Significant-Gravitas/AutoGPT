# MrScraper Get Screenshot
<!-- MANUAL: file_description -->
Captures a PNG screenshot of a page. Needs a MrScraper API key: create one at [app.mrscraper.com/api-tokens](https://app.mrscraper.com/api-tokens) and add it as a MrScraper credential.
<!-- END MANUAL -->

## MrScraper Get Screenshot

### What it is
Capture a base64 PNG screenshot of the full page, the first viewport, or a custom viewport. Browser rendering is always enabled.

### How it works
<!-- MANUAL: how_it_works -->
The block sends `GET https://api.mrscraper.com/` with `browserRendering=true`, `html=false`, `markdown=false` and `saveResult=true`; browser rendering is always on. `screenshot` is `full` (the whole scroll height) unless you set `top` (the first viewport) or a `WIDTHxHEIGHT` viewport such as `1280x800`, with an optional zoom such as `1920x1080-75`. MrScraper returns a base64 PNG, which the block saves to your files and outputs as `screenshot`. The API also returns the page HTML, which the block outputs as `html`. The request waits up to `timeout` plus 30 seconds and is sent once.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | Absolute HTTP or HTTPS URL of the page to process. | str | Yes |
| screenshot | Screenshot area to capture: full for the entire scroll height, top for the first viewport, or a WIDTHxHEIGHT viewport from 100 to 3840 pixels with an optional -ZOOM percent from 25 to 300, such as 1280x800 or 1920x1080-75. | str | No |
| timeout | Maximum seconds to wait for the scrape, from 1 to 330. The HTTP request allows 30 additional seconds. | int | No |
| wait_for_selector | CSS selector or text to wait for on the page before returning the result, up to 500 characters. | str | No |
| geo_code | Two-letter country code to load the page from, sent in lowercase. | str | No |
| max_age | Return a cached result of the same request when it is newer than the selected age. A cached result costs the same tokens as the original scrape. | int | No |
| max_retries | Maximum number of retries, as a nonnegative integer. Omit this parameter for no limit. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| screenshot | PNG screenshot of the page. | str (file) |
| html | HTML of the page, which the API includes with the screenshot. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Visual monitoring**: Capture a page on a schedule and send the image when something changes.

**Reports**: Add screenshots of competitor or product pages to a generated report.

**Check a layout at a size**: Use a viewport such as `390x844` or `1440x900` to see how a page renders at that width.
<!-- END MANUAL -->

---
