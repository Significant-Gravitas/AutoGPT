# Google Search Console
<!-- MANUAL: file_description -->
Blocks for reading a site's Google Search traffic from Google Search Console: list the properties (sites) the connected Google account can see, and get clicks, impressions, CTR and average position by query, page, country, device or date. They only read, with the `webmasters.readonly` scope.

Every Search Console block takes a `site_url`: a property exactly as Search Console lists it, `sc-domain:example.com` for a domain property or `https://www.example.com/` for a URL-prefix property (a missing trailing slash is added). A bare domain such as `example.com` works too. The block then lists the account's properties once and takes the first one it can read, in this order: the domain property, then the `https://` URL-prefix property, then the `http://` one. `example.com` and `www.example.com` count as the same site, with the form given tried first. If nothing matches, the error lists the properties the account can read. The property used comes out as `site_url`.

Setup: the Google Search Console API must be enabled in the Google Cloud project behind AutoGPT's Google sign-in, and the connected Google account needs at least restricted access to the property. An owner adds people in Search Console under Settings > Users and permissions.
<!-- END MANUAL -->

## Google Search Console Get Performance

### What it is
Get a site's search performance from Google Search Console: clicks, impressions, CTR and average position, by query, page, country, device or date. This is the Search Analytics data behind Search Console's Performance report, for SEO questions such as a website's top search queries (keywords), its top pages and how they rank. It can filter rows and also report on image, video, news, Discover and Google News results.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Search Console API `searchanalytics.query` endpoint, which serves the data behind Search Console's Performance report. Each row holds the dimensions you grouped by, in the order you gave them, plus clicks, impressions, CTR (0 to 1) and average position (1 is the top). With no dimensions you get one row of totals. Rows come most clicks first, or oldest first when grouped by date, and days without data are left out. Dates take the forms Google Analytics uses (`YYYY-MM-DD`, `today`, `yesterday`, `NdaysAgo`) and count in Pacific Time, as Search Console does. Its final numbers usually arrive 2–3 days late, so the default range is the 28 days from `30daysAgo` to `3daysAgo`. To include the latest days, set `end_date` to `today` and turn on `include_fresh_data`; those numbers are preliminary and can still change. Search Console keeps 16 months of data.

Filters are combined with AND, the only grouping Google supports. `contains` ignores case, `equals` is case-sensitive for queries and pages, and the regex operators take RE2 patterns. Discover and Google News have no query dimension and report no position; Google rejects an unsupported dimension and the block passes its message on. A request returns up to 25,000 rows (`row_limit`), and `start_row` pages through more. Google doesn't promise every row: it serves at most 50,000 rows a day per search type, grouping by page or query can drop some data, and rare queries are left out for privacy, so query rows usually add up to less than the totals. Each property and each user can make 1,200 requests a minute, and long date ranges grouped by both page and query use up Google's load quota fastest.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| site_url | The Search Console property, as Search Console lists it: sc-domain:example.com for a domain property or https://www.example.com/ for a URL-prefix property. A bare domain like example.com also works: the block picks the matching property the account can read. | str | Yes |
| start_date | First day to count: YYYY-MM-DD, today, yesterday or NdaysAgo, in Pacific Time like Search Console | str | No |
| end_date | Last day to count, in the same forms. Search Console's final numbers run 2-3 days behind, so the default is 3daysAgo. | str | No |
| dimensions | What to break the numbers down by, in this order. Leave empty for one row of totals. | List["query" \| "page" \| "country" \| "device" \| "date" \| "searchAppearance"] | No |
| search_type | Which results to count: web (the main results), image, video, news (the News tab), discover, or googleNews (news.google.com and the Google News app). Discover and Google News have no query dimension and no position. | "web" \| "image" \| "video" \| "news" \| "discover" \| "googleNews" | No |
| filters | Only count rows that meet every one of these conditions, such as query contains 'shoes' | List[SearchConsoleFilter] | No |
| row_limit | Most rows to return (Google allows up to 25,000) | int | No |
| start_row | Rows to skip, to page through more rows than row_limit | int | No |
| include_fresh_data | Also count the last few days, whose numbers aren't final yet and can still change. Set end_date to today or yesterday to see them. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| rows | The rows, most clicks first (oldest first when grouped by date) | List[SearchConsoleRow] |
| row | Each row | SearchConsoleRow |
| site_url | The property the numbers are for | str |

### Possible use case
<!-- MANUAL: use_case -->
**Weekly SEO Report**: Every Monday, pull the top queries and pages of the last 28 days and email a summary of what changed.

**Quick Wins**: Find queries with many impressions where the site ranks between 5 and 15, and suggest which pages to improve.

**Traffic Drop Alerts**: Compare clicks by page for this month and the last, and flag the pages that lost the most.
<!-- END MANUAL -->

---

## Google Search Console List Sites

### What it is
List the Google Search Console properties (sites) the connected Google account can see, with its permission level for each. Pass a property's site_url to the other Search Console blocks.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Search Console API `sites.list` endpoint. It returns every property the connected Google account can see, each with the account's permission level: `siteOwner`, `siteFullUser`, `siteRestrictedUser` or `siteUnverifiedUser`. An unverified user is listed but can't read the property's data, so the other Search Console blocks skip such properties when they match a bare domain.

Domain properties (`sc-domain:example.com`) cover every subdomain and protocol of the domain. URL-prefix properties (`https://www.example.com/`) cover only URLs that start with that prefix. Pass `site_url` exactly as listed to the other Search Console blocks.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| sites | Every property the account can see | List[SearchConsoleSite] |
| site | Each property | SearchConsoleSite |

### Possible use case
<!-- MANUAL: use_case -->
**Pick the Right Property**: Find the exact property name to give Get Performance, Inspect URL or List Sitemaps.

**Agency Reports**: Run a monthly performance report for every property the connected account can read.

**Access Check**: Confirm the connected account can read a site's data before an agent relies on it.
<!-- END MANUAL -->

---
