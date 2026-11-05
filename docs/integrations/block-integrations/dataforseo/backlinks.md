# Dataforseo Backlinks
<!-- MANUAL: file_description -->
Blocks for reading a site's backlink profile and the sites linking to it, from DataForSEO Backlinks.
<!-- END MANUAL -->

## Data For Seo Backlinks Summary

### What it is
Get a domain's or page's backlink profile from DataForSEO: authority rank, backlinks, referring domains, broken backlinks and spam score

### How it works
<!-- MANUAL: how_it_works -->
This block calls the DataForSEO Backlinks Summary API and returns a target's authority rank (0-1000), live backlinks, referring domains, referring domains with nofollow links only, broken backlinks, broken pages and spam score (0-100). It works for any domain, subdomain or page, including competitors.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| target | Domain without https:// or www. (e.g. 'example.com'), a subdomain, or a full page URL. Works for any site, including competitors | str | Yes |
| include_subdomains | Count links to the target's subdomains too | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| summary | Live backlinks and referring domains. rank: DataForSEO's authority score, 0-1000 (higher is stronger); comparable to other tools' domain ratings but not the same scale. broken_backlinks are links pointing at pages that no longer load; spam_score is 0-100 | BacklinksSummary |

### Possible use case
<!-- MANUAL: use_case -->
**Link gap**: Compare your referring domains with a competitor's to see which sites link to them but not to you, for outreach and digital PR.

**Broken backlinks**: See how many links point at pages that no longer load, a sign that old URLs need redirects.

**Authority check**: Judge how hard it will be to outrank the pages already ranking for a query.
<!-- END MANUAL -->

---

## Data For Seo Referring Domains

### What it is
List the sites linking to a domain or page from DataForSEO Backlinks, strongest first, with rank, link count and spam score

### How it works
<!-- MANUAL: how_it_works -->
This block calls the DataForSEO Referring Domains API and lists the sites with live links to the target, highest authority rank first, excluding the target's own internal links. Each comes with its rank, how many links it has to the target, its spam score and when the first link was seen. `total_count` gives the full number of referring domains beyond the limit.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| target | Domain without https:// or www. (e.g. 'example.com'), a subdomain, or a full page URL. Works for any site, including competitors | str | Yes |
| include_subdomains | Count links to the target's subdomains too | bool | No |
| limit | Maximum referring domains to return, highest rank first | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| referring_domains | Sites with live links to the target, highest rank first. rank: DataForSEO's authority score, 0-1000 (higher is stronger); comparable to other tools' domain ratings but not the same scale. backlinks is how many links that site has to the target; spam_score is 0-100 | List[ReferringDomain] |
| total_count | How many referring domains there are in total | int |

### Possible use case
<!-- MANUAL: use_case -->
**Link gap**: Compare your referring domains with a competitor's to see which sites link to them but not to you, for outreach and digital PR.

**Broken backlinks**: See how many links point at pages that no longer load, a sign that old URLs need redirects.

**Authority check**: Judge how hard it will be to outrank the pages already ranking for a query.
<!-- END MANUAL -->

---
