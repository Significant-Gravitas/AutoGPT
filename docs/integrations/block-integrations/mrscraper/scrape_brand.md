# MrScraper Scrape Brand
<!-- MANUAL: file_description -->
Extracts a website's brand: colors, typography, logos and social links. Needs a MrScraper API key: create one at [app.mrscraper.com/api-tokens](https://app.mrscraper.com/api-tokens) and add it as a MrScraper credential.
<!-- END MANUAL -->

## MrScraper Scrape Brand

### What it is
Extract a website's colors, typography, logos, and social links as JSON.

### How it works
<!-- MANUAL: how_it_works -->
The block sends `GET https://api.mrscraper.com/` with `action=brand-scraping`, the URL and `bundle=none`. The call has a fixed 120-second deadline and takes no page controls. The block fails when MrScraper reports an error or a job status other than `succeeded`. `brand` holds the brand details, `final_url` is the page after redirects, and `response` is the raw body.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | Absolute HTTP or HTTPS URL of the page to process. | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| brand | Brand details: name, colors, typography, logos, and social links. | Dict[str, Any] |
| final_url | URL of the page after redirects. | str |
| response | Raw brand scraping API response, with message and data. | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**On-brand content**: Pass `brand.colors` and the typography into an image, slide or email generation step.

**Lead enrichment**: Add a company's logo and social links to a CRM record.

**Competitor research**: Collect the brand details of several sites in one workflow.
<!-- END MANUAL -->

---
