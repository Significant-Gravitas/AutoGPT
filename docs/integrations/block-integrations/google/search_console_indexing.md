# Google Search Console Indexing
<!-- MANUAL: file_description -->
Blocks for checking how Google crawls and indexes a site, from Google Search Console: inspect one URL's index status, and list the sitemaps submitted for a property. They only read, with the `webmasters.readonly` scope. `site_url` and the setup work as for the other Search Console blocks (see [Google Search Console](search_console.md)): the Google Search Console API enabled in the Google Cloud project behind AutoGPT's Google sign-in, and at least restricted access to the property.
<!-- END MANUAL -->

## Google Search Console Inspect URL

### What it is
Inspect a URL with Google Search Console: whether Google has indexed it and if not why, when it was last crawled, its canonical and any rich results. It reports the version in Google's index and can't test the live page.

### How it works
<!-- MANUAL: how_it_works -->
Calls the URL Inspection API `urlInspection.index.inspect` endpoint, the data behind Search Console's URL Inspection tool. It reports the version of the page in Google's index and can't run a live test. The URL must be inside the property; with a bare domain as `site_url`, only properties that contain the URL count. `verdict` is `PASS` for an indexed page, `NEUTRAL` for one Search Console counts as excluded (by noindex, as a redirect or as a duplicate, for example) and `FAIL` for an error, and `coverage_state` gives the reason in Search Console's words. Google allows 2,000 inspections a day and 600 a minute for each property.

Google leaves out what doesn't apply, such as the canonical of a page it hasn't indexed or rich results for a page without any, and so does the block. The three lists always come out, empty if need be. The full result, with AMP details and rich result issues, is in `inspection_result`. Google retired its Mobile Usability report on December 1, 2023 and marks the API's mobile usability result deprecated, so there's no output for it; if Google still sends one, it stays in `inspection_result`. `language_code` sets the language of Google's issue messages.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inspection_url | The full URL of the page to inspect, inside the property | str | Yes |
| site_url | The Search Console property, as Search Console lists it: sc-domain:example.com for a domain property or https://www.example.com/ for a URL-prefix property. A bare domain like example.com also works: the block picks the matching property the account can read. For a bare domain, only properties that contain the URL count. | str | Yes |
| language_code | Language for Google's issue messages, as a BCP-47 code such as en-US or de-CH | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| verdict | Google's verdict on the page: PASS (indexed), NEUTRAL (excluded, for example by noindex or as a duplicate) or FAIL (an error) | str |
| coverage_state | Why the page is or isn't indexed, in Search Console's words, such as 'Submitted and indexed' or 'Crawled - currently not indexed' | str |
| indexing_state | INDEXING_ALLOWED, or BLOCKED_BY_META_TAG or BLOCKED_BY_HTTP_HEADER when a noindex rule blocks it | str |
| robots_txt_state | ALLOWED or DISALLOWED by the site's robots.txt | str |
| page_fetch_state | Whether Google could fetch the page: SUCCESSFUL, or a problem such as SOFT_404, NOT_FOUND, SERVER_ERROR or REDIRECT_ERROR | str |
| last_crawl_time | When Google last crawled the page (RFC 3339, UTC), if ever | str |
| crawled_as | The crawler Google used: MOBILE or DESKTOP | str |
| google_canonical | The URL Google picked as canonical, once the page is indexed | str |
| user_canonical | The canonical URL the page declares, if it declares one | str |
| sitemaps | Sitemaps that Google knows list the URL (not always all of them) | List[str] |
| referring_urls | Pages that Google knows link to the URL | List[str] |
| rich_results_verdict | PASS or FAIL for the page's rich results, if it has any | str |
| rich_result_types | The kinds of rich result found, such as Breadcrumbs or FAQ | List[str] |
| inspection_result_link | Link to the URL's report in Search Console | str |
| inspection_result | Google's full inspection result, including AMP and rich result issues | Dict[str, Any] |
| site_url | The property the URL was inspected in | str |

### Possible use case
<!-- MANUAL: use_case -->
**Check New Pages**: A few days after a post goes live, check that Google indexed it and tell the author if not.

**Canonical Audit**: For key pages, compare the canonical Google picked with the one the page declares and flag mismatches.

**Traffic Drop Triage**: For pages that lost clicks in Get Performance, check whether they're still indexed and when Google last crawled them.
<!-- END MANUAL -->

---

## Google Search Console List Sitemaps

### What it is
List a site's sitemaps in Google Search Console: when Google last read each one, its errors and warnings, and how many URLs it lists. It can also list the sitemaps inside a sitemap index.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Search Console API `sitemaps.list` endpoint. It returns the sitemaps submitted for the property or, with `sitemap_index`, the sitemaps listed in that sitemap index. Each comes with its type, when it was last submitted and downloaded, whether Google has yet to process it, its error and warning counts, and how many pages, images or videos it lists.

The API also has an indexed count for each kind of content, but Google marks it deprecated ("do not use"), so the block leaves it out. To check whether a page is indexed, use Google Search Console Inspect URL.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| site_url | The Search Console property, as Search Console lists it: sc-domain:example.com for a domain property or https://www.example.com/ for a URL-prefix property. A bare domain like example.com also works: the block picks the matching property the account can read. | str | Yes |
| sitemap_index | The full URL of a sitemap index, to list the sitemaps inside it instead | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| sitemaps | The sitemaps, with what Google made of each | List[SearchConsoleSitemap] |
| sitemap | Each sitemap | SearchConsoleSitemap |
| site_url | The property the sitemaps belong to | str |

### Possible use case
<!-- MANUAL: use_case -->
**Sitemap Health Alerts**: Check the sitemaps every day and alert when one has errors or Google hasn't downloaded it lately.

**Launch Checks**: After a site launch or migration, confirm Google processed the new sitemap and that it lists the expected number of pages.

**Sitemap Index Review**: List the child sitemaps of a large site's sitemap index and find the ones with warnings.
<!-- END MANUAL -->

---
