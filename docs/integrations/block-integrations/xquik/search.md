# Xquik Search
<!-- MANUAL: file_description -->
Blocks for searching public X posts through Xquik.
<!-- END MANUAL -->

## Xquik Search Tweets

### What it is
Searches public X posts through Xquik without requiring an X developer account

### How it works
<!-- MANUAL: how_it_works -->
Add an Xquik API key, then provide words, hashtags, a username, post ID, or
post URL. The block accepts a `limit` from 1 through 100. It omits unset
filters when calling the official Xquik Python SDK. Supported public reads do
not require an X developer account or a connected X account.

Use the date, language, author, and sort fields to narrow the search. The block
returns one typed list and emits each post separately for downstream steps.
When `has_next_page` is true, pass `next_cursor` into another run. The block
omits `next_cursor` when no page follows. AutoGPT's block executor adds block
context to SDK errors.

Xquik is an independent third-party service. It is not affiliated with X Corp.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Words, hashtags, usernames, a post ID, or an X post URL | str | Yes |
| limit | Maximum posts to return | int | No |
| sort_order | Sort by newest posts or engagement | "Latest" \| "Top" | No |
| language | Language code such as en or tr | str | No |
| from_user | Only return posts from this username | str | No |
| since_date | Earliest post date | str (date) | No |
| until_date | Latest post date | str (date) | No |
| cursor | Cursor returned by the previous search | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| tweets | Matching X posts | List[XquikTweetResult] |
| tweet | One matching X post | XquikTweetResult |
| next_cursor | Cursor for the next page | str |
| has_next_page | Whether another result page is available | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Historical research.** Search tweets by date, author, or keyword. Send each
result to an analysis or storage block.

**Mention tracking.** Run a recurring Twitter search for a product name. Use
the cursor to continue without rebuilding the query.

**Competitive research.** Find public X posts about a market or competitor.
Route each result into an analysis workflow.
<!-- END MANUAL -->

---
