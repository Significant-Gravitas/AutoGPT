# Dataforseo Ranked Keywords
<!-- MANUAL: file_description -->
Blocks for finding the Google searches a site or page ranks for, from DataForSEO Labs.
<!-- END MANUAL -->

## Data For Seo Ranked Keywords

### What it is
Get the Google keywords a domain or page ranks for from DataForSEO Labs, with position, ranking URL, search volume and estimated traffic

### How it works
<!-- MANUAL: how_it_works -->
This block calls the DataForSEO Labs Ranked Keywords API for a domain or page URL and returns the keywords it ranks for in Google, highest estimated traffic first. Each keyword comes with its organic position, the ranking URL, monthly search volume, cost per click, keyword difficulty and search intent.

Set `min_position` and `max_position` to keep only a range, e.g. 7 to 15 for keywords close to the top of page one. The data comes from DataForSEO's regularly refreshed database rather than a live search, and works for any site, including competitors.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| target | Domain without https:// or www. (e.g. 'example.com') for the whole site, or a full page URL for one page. Works for any site, including competitors | str | Yes |
| location_code | Location code for targeting (e.g., 2840 for USA) | int | No |
| language_code | Language code (e.g., 'en' for English) | str | No |
| min_position | Only keywords ranking at this position or lower, e.g. 7 | int | No |
| max_position | Only keywords ranking at this position or higher, e.g. 15 | int | No |
| limit | Maximum keywords to return, highest estimated traffic first | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| keywords | Ranked keywords. position is the organic position (1 = first organic result); estimated_traffic is DataForSEO's monthly visit estimate for that position, not measured visits | List[RankedKeyword] |
| total_count | How many keywords match in total, beyond the limit | int |

### Possible use case
<!-- MANUAL: use_case -->
**Striking distance**: Find keywords a site ranks 7 to 15 for, where a page improvement can reach the top three.

**Competitor gaps**: List a competitor's non-branded keywords to find demand your site has no page for.

**Cannibalisation**: Spot keywords where more than one of your URLs ranks.
<!-- END MANUAL -->

---
