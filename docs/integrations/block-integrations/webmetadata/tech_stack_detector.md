# Webmetadata Tech Stack Detector
<!-- MANUAL: file_description -->
Blocks backed by the [Web Metadata Extractor API](https://webmetadataextractor.com/) (free tier: 1,000 requests/month, no card required).
<!-- END MANUAL -->

## Tech Stack Detector

### What it is
Fingerprints the CMS/framework/analytics stack (WordPress, Shopify, Next.js, etc.) a URL is built on.

### How it works
<!-- MANUAL: how_it_works -->
The block sends the target URL to the API's `/api/v1/tech-stack` endpoint, which fetches the page once and matches its response headers and HTML markers against 40+ known technology signatures (CMS, e-commerce, frameworks, analytics, hosting/CDN). Each match is returned with a category (e.g. `cms`, `analytics`), a confidence score, and the specific evidence string that triggered the detection (e.g. `wp-content` for WordPress).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The URL to inspect | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| detected_technologies | Names of every technology detected | List[str] |
| technology | A single detected technology, with category/confidence/evidence | TechnologyDetail |
| technologies | All detected technologies, with category/confidence/evidence | List[TechnologyDetail] |

### Possible use case
<!-- MANUAL: use_case -->
**Lead Qualification by Platform**: Have an agent scan a list of prospect URLs and route Shopify sites to one outreach template and WordPress sites to another.

**Integration Feasibility Check**: Before proposing an integration with a target site, detect whether it's built on a platform (e.g. Webflow) that would block or complicate the approach.

**Competitive Landscape Mapping**: Loop this block over a category of competitor sites to build a quick report of which frameworks/analytics tools are common in that market.
<!-- END MANUAL -->

---
