# Dataforseo Serp
<!-- MANUAL: file_description -->
Blocks for fetching live Google search results pages from DataForSEO.
<!-- END MANUAL -->

## Data For Seo Google Search Results

### What it is
Get live Google search results for a keyword from DataForSEO: organic rankings, AI Overview, People also ask and related searches

### How it works
<!-- MANUAL: how_it_works -->
This block calls the DataForSEO SERP API (Google Organic, live advanced mode) for one keyword, location, language and device, and returns what a searcher there sees. Organic results are numbered by organic position only, so ads, AI Overviews and other features don't shift the count. It also returns the AI Overview as markdown, the "People also ask" questions and the related searches.

Google now links results through an opaque `google.com/goto` redirect. DataForSEO normally resolves these to the real page; when it can't, the result's `url_resolved` is false and the domain is taken from the result's breadcrumb. Billing is per 10 results fetched (`depth`).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| keyword | The search query to look up | str | Yes |
| location_code | Location code for targeting (e.g., 2840 for USA) | int | No |
| language_code | Language code (e.g., 'en' for English) | str | No |
| device | Which device's results to fetch | "desktop" \| "mobile" | No |
| depth | How many results to fetch (10 per page; billed per 10) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| organic_results | Unpaid results in order. position counts organic listings only (1 = first organic result). url_resolved is false when Google's redirect link could not be resolved; domain then comes from the result's breadcrumb | List[OrganicResult] |
| ai_overview | Google's AI Overview as markdown, or empty if none showed | str |
| people_also_ask | Questions in the 'People also ask' box | List[str] |
| related_searches | Searches Google suggests at the bottom of the page | List[str] |
| result_types | Every kind of result on the page, e.g. organic, video, local_pack | List[str] |
| check_url | Google URL that shows the same results in a browser | str |

### Possible use case
<!-- MANUAL: use_case -->
**Rank checks**: See where a site appears for a target search today, and who outranks it.

**Content briefs**: Read the page types, People also ask questions and AI Overview that Google shows for a query before writing for it.

**Search intent**: Check whether a query rewards product, category, guide or comparison pages.
<!-- END MANUAL -->

---
