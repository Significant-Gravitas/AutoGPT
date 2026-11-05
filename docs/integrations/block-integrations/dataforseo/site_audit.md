# Dataforseo Site Audit
<!-- MANUAL: file_description -->
Blocks for crawling a website and reporting its technical SEO issues, with DataForSEO On-Page.
<!-- END MANUAL -->

## Data For Seo Site Audit

### What it is
Crawl a website with DataForSEO On-Page and report technical SEO issues per page: broken pages, redirects, titles, descriptions, H1s, canonicals, duplicates, orphan pages, speed and click depth

### How it works
<!-- MANUAL: how_it_works -->
This block starts a DataForSEO On-Page crawl of a domain (up to `max_crawl_pages` pages), waits for it to finish (up to `max_wait_seconds`), then returns the site's on-page score, how many pages have each issue, and every crawled page with its title, description, H1s, canonical, click depth from the home page and issues, lowest-scoring first.

Issues use DataForSEO's check names, keeping only the ones that signal a problem: for example `is_4xx_code`, `is_redirect`, `no_title`, `title_too_long`, `no_description`, `no_h1_tag`, `canonical_to_redirect`, `is_orphan_page`, `no_image_alt`, `has_micromarkup_errors`, `duplicate_title` and `broken_links`. If the crawl is still running when the wait ends, the pages crawled so far are returned with `finished` false; run the block again with the same `task_id` to collect the rest without paying for a new crawl. Crawls are billed per page.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| target | Domain to crawl, without https:// or www. (e.g. 'example.com') | str | Yes |
| max_crawl_pages | Most pages to crawl (billed per page crawled) | int | No |
| max_wait_seconds | How long to wait for the crawl. If it is still running, the pages crawled so far are returned; pass task_id to collect the rest later | int | No |
| task_id | Collect the results of an earlier crawl instead of starting one | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| task_id | The crawl's ID, kept for 30 days | str |
| finished | Whether the crawl has finished | bool |
| pages_crawled | How many pages were crawled | int |
| onpage_score | Site score, 0-100, where 100 means no critical issues | float |
| issue_counts | How many crawled pages have each issue, most common first | Dict[str, int] |
| pages | Crawled pages, lowest score first, each with its issues (On-Page check names such as no_h1_tag, is_4xx_code, duplicate_title) and click_depth: clicks from the home page | List[AuditedPage] |

### Possible use case
<!-- MANUAL: use_case -->
**Technical SEO audit**: Find broken pages, redirect chains, missing titles, descriptions and H1s, and canonical problems across a site.

**Site structure**: See pages buried many clicks from the home page, or orphan pages nothing links to.

**Before and after**: Re-crawl after a fix or migration to confirm issues are gone.
<!-- END MANUAL -->

---
