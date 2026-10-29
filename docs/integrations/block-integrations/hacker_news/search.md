# Hacker News Search
<!-- MANUAL: file_description -->
Search Hacker News stories and comments through HN Search (hn.algolia.com), the public search API that Algolia runs for Hacker News. It needs no account or API key. HN Search allows 10,000 requests an hour from one IP address, and every agent running on the same server shares that allowance.
<!-- END MANUAL -->

## Hacker News Search

### What it is
Search Hacker News stories and comments by keyword, author, date, points or linked domain, sorted by relevance or newest first. Use it to find mentions of a product, links to a website, or the most upvoted stories of the week. Uses HN Search by Algolia, which needs no account. exact_match is on by default, so words that are only spelled alike are left out; turn it off to also catch misspellings.

### How it works
<!-- MANUAL: how_it_works -->
Calls HN Search's `search` endpoint, or `search_by_date` when `sort_by` is newest, and gets up to 1,000 results in one request. Relevance puts the best matches first, then the stories with the most points, then the most comments. With an empty query everything matches equally, so relevance sorts by points. `search_for` becomes HN Search's `tags` filter (`story`, `comment`, `(story,comment)`, `show_hn`, `ask_hn` or `front_page`), plus `author_<username>` when `author` is set. The time inputs and the minimums become `numericFilters` on `created_at_i`, `points` and `num_comments`. Only stories have points and comment counts, so a minimum above 0 leaves comments out. `match_in` limits matching to a story's `title` or `url`. Comments have neither, so the block refuses that combination instead of returning nothing.

HN Search allows typos unless told not to. Then `autogpt` also matches `automotive` and `automation`, hundreds of thousands of comments instead of a few hundred, and sorted newest first those near misses take over. So `exact_match` is on by default and turns typo tolerance off. Plurals still match, and the last word still matches the start of longer words (`agent` finds `agents` and `agentic`). Turn `exact_match` off to also catch misspellings. Quotes around a phrase and a `-` before a word work too. `total_matches` is HN Search's count of every match, even past the 1,000 it returns. Text comes back as plain text: paragraphs become blank lines, italics become `*asterisks*`, code keeps its indentation, and links show their full URL. HN Search adds new stories and comments a minute or two after they are posted. Times are in UTC.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Words to search for. Put a phrase in double quotes to match it as a phrase, and put - before a word to leave out items that have it. Leave empty to match everything, e.g. for the top stories of the past week. | str | No |
| search_for | What to search: stories, comments, stories_and_comments, show_hn or ask_hn posts, or front_page (only the stories on the front page now) | "stories" \| "comments" \| "stories_and_comments" \| "show_hn" \| "ask_hn" \| "front_page" | No |
| match_in | Where the query must match: anywhere (title, link, text and author), title, or url (the link a story points to, e.g. to find links to a domain). title and url only match stories. | "anywhere" \| "title" \| "url" | No |
| exact_match | Match the query words as written, without typo tolerance. On by default, because typo tolerance makes 'autogpt' also match 'automotive' and 'automation'. Plurals still match. Turn it off to also catch misspellings. | bool | No |
| sort_by | relevance: best matches first, then most points, then most comments. newest: newest first. | "relevance" \| "newest" | No |
| created_after | Only items posted at or after this time: an ISO 8601 date or time (2026-09-01, 2026-09-01T12:00:00Z; UTC unless it has an offset), or an age such as 24h, 7d or 2w | str | No |
| created_before | Only items posted before this time, written the same way as created_after | str | No |
| author | Only items posted by this Hacker News username (case-sensitive) | str | No |
| min_points | Only stories with at least this many points. Comments have no points, so any value above 0 leaves comments out. | int | No |
| min_comments | Only stories with at least this many comments. Any value above 0 leaves comments out. | int | No |
| max_results | Most results to return (up to 1,000) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| results | Matching stories and comments, in the chosen order | List[HackerNewsItem] |
| result | Each matching story or comment | HackerNewsItem |
| total_matches | How many items match in all. Can be more than the results returned: HN Search returns at most 1,000. | int |

### Possible use case
<!-- MANUAL: use_case -->
**Brand Monitoring**: Every morning, search stories and comments from the past day for your product's name, newest first, and post new mentions to Slack.

**Who Links to Us**: Set `match_in` to `url` and search for your domain to find every story that links to your site, with its points and comment count.

**Weekly Digest**: Leave the query empty, set `created_after` to `7d` and sort by relevance to get the most upvoted stories of the week.
<!-- END MANUAL -->

---
