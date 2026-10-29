# Hacker News Stories
<!-- MANUAL: file_description -->
Get Hacker News's live story lists: the front page, new, best, Ask HN, Show HN and jobs. The block calls Hacker News's official API (hacker-news.firebaseio.com), which needs no account or API key and publishes no rate limit.
<!-- END MANUAL -->

## Hacker News Get Stories

### What it is
Get the top (front page), new, best, Ask HN, Show HN or job stories from Hacker News, in the order Hacker News ranks them. Each story comes with its rank, title, link, points, comment count and poster. Uses Hacker News's official API, which needs no account.

### How it works
<!-- MANUAL: how_it_works -->
Reads the list's ids from the official API (`topstories`, `newstories`, `beststories`, `askstories`, `showstories` or `jobstories`), keeps the first `max_results` and fetches each story, 10 at a time. The official API returns one item per request, so 500 stories take 501 requests and a few seconds. Stories stay in the list's order, and `rank` is each story's place on the list. Stories deleted or killed since the list was made are left out, so a run can return a few fewer than `max_results`, and the ranks skip their places.

Top, new and best hold up to 500 stories; Ask HN, Show HN and jobs up to 200, and often far fewer. Jobs have no points or comments. The text of Ask HN and other text posts comes back as plain text, and times are in UTC.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| story_list | Which list: top (the front page ranking), new, best (most upvoted lately), ask (Ask HN), show (Show HN) or jobs | "top" \| "new" \| "best" \| "ask" \| "show" \| "jobs" | No |
| max_results | Most stories to return, from the top of the list. Top, new and best hold up to 500 stories; ask, show and jobs up to 200. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| stories | The stories, in the list's order | List[HackerNewsStory] |
| story | Each story | HackerNewsStory |

### Possible use case
<!-- MANUAL: use_case -->
**Front Page Digest**: Get the top 30 stories every morning and send a summary to Slack or email.

**Launch Watch**: Check the Show HN list for launches in your market and alert the team.

**Hiring Signals**: Read the jobs list to see which YC companies are hiring, and for what roles.
<!-- END MANUAL -->

---
