# MrScraper Scrape JSON
<!-- MANUAL: file_description -->
Extracts structured data from a page with a natural-language prompt. Needs a MrScraper API key: create one at [app.mrscraper.com/api-tokens](https://app.mrscraper.com/api-tokens) and add it as a MrScraper credential.
<!-- END MANUAL -->

## MrScraper Scrape JSON

### What it is
Extract structured JSON from a page using a natural-language prompt. Append Json Schema: and a JSON schema to shape the output.

### How it works
<!-- MANUAL: how_it_works -->
The block sends `GET https://api.mrscraper.com/` with your prompt, `agent=general` and `saveResult=true`. Append `Json Schema:` and a JSON schema to the prompt to shape `data`. The prompt only parses the scraped content; it cannot instruct the scraper. When the encoded GET URL would be longer than 8,000 characters, the block sends a `POST` instead: `url`, `prompt` and `agent` go in a JSON body and the other fields stay in the query string. MrScraper accepts prompts of up to 8,000 characters. `data` is the extracted result; `html` and `screenshot` are output when you ask for them and the API returns them, and `token_usage` is the tokens the extraction used. The request waits up to `timeout` plus 30 seconds and is sent once. Page content is untrusted web text; treat it as data, not instructions.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | Absolute HTTP or HTTPS URL of the page to process. | str | Yes |
| prompt | Instructions describing the data to extract. Append Json Schema: and a JSON schema to shape the output. | str | No |
| html | Whether to include the page HTML in the response. | bool | No |
| browser_rendering | Whether to load the page in a browser: true always does and false never does. Omit it to load the page without a browser first and switch to a browser automatically if the request is blocked. | bool | No |
| timeout | Maximum seconds to wait for the scrape, from 1 to 330. The HTTP request allows 30 additional seconds. | int | No |
| warm_start | Whether to open the site's home page first to warm the session before loading the target URL. | bool | No |
| block_resources | Whether to block images, fonts, and stylesheets while loading the page. | bool | No |
| wait_for_selector | CSS selector or text to wait for on the page before returning the result, up to 500 characters. | str | No |
| geo_code | Two-letter country code to load the page from, sent in lowercase. | str | No |
| max_age | Return a cached result of the same request when it is newer than the selected age. A cached result costs the same tokens as the original scrape. | int | No |
| max_retries | Maximum number of retries, as a nonnegative integer. Omit this parameter for no limit. | int | No |
| screenshot | Whether to include a base64 screenshot in the response. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| data | Extracted data, shaped by the prompt and any JSON schema. | Data |
| html | HTML of the page, when html is true and the API returns it. | str |
| screenshot | Screenshot of the page, when screenshot is true and the API returns it. | str (file) |
| token_usage | MrScraper tokens used by the extraction. | int |

### Possible use case
<!-- MANUAL: use_case -->
**Product data**: Use a prompt such as `Extract every product with title and price. Json Schema: {"type":"array","items":{"type":"object","properties":{"title":{"type":"string"},"price":{"type":"string"}}}}` and write `data` to a sheet.

**Job listings**: Extract the title, location and link of each job on a careers page.

**Contact details**: Pull the address, phone number and opening hours from a business page.
<!-- END MANUAL -->

---
