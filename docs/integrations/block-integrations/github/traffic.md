# GitHub Traffic
<!-- MANUAL: file_description -->
A block for reading a repository's traffic: the views, git clones, referring sites and popular content on its **Insights > Traffic** page. GitHub only shows traffic to people with push (write) access to the repository. The GitHub connection or a classic personal access token needs the `repo` scope, and a fine-grained token needs the **Administration** repository permission set to read.
<!-- END MANUAL -->

## Github Get Repository Traffic

### What it is
Get a GitHub repository's views and clones over the last 14 days, with unique visitors and cloners, plus its top referring sites and most viewed pages. Views and clones are also broken down by day or week. GitHub only shows traffic to accounts with push access to the repository.

### How it works
<!-- MANUAL: how_it_works -->
Calls GitHub's four traffic endpoints at the same time: `traffic/views` and `traffic/clones` (with `per` set to `day` or `week`), `traffic/popular/referrers` and `traffic/popular/paths`. `repo_url` can be a github.com link or `owner/repo`. Anything else is rejected before a request is made. If the connected account can't push to the repository, or a fine-grained token lacks the Administration (read) permission, GitHub refuses and the block says which access is missing. A repository that doesn't exist, or that the account can't see, fails as not found. Other errors, such as rate limits, keep GitHub's own message.

GitHub keeps only the last 14 days. Days and weeks start at midnight UTC, and weeks start on Monday. Views and clones refresh every hour; referring sites and popular content refresh once a day. Clones are full git clones, not fetches. Unique counts don't add up across days: someone who visited on three days counts once in `unique_visitors` but once on each of those days in `views_by_period`. `referrers` and `popular_content` are GitHub's top 10 for the 14 days, and each page comes with its github.com link. GitHub lists each path as it was visited, so the same page can appear twice with different capitalisation.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | Repository URL or '{owner}/{repo}' | str | Yes |
| period | Whether to break the views and clones down by day or by week. Days and weeks start at midnight UTC, and weeks start on Monday. | "day" \| "week" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| views | Views in the last 14 days | int |
| unique_visitors | Unique visitors in the last 14 days. Someone who visited on several days counts once. | int |
| views_by_period | Views and unique visitors for each day or week, oldest first | List[TrafficCount] |
| clones | Git clones in the last 14 days. Counts full clones, not fetches. | int |
| unique_cloners | Unique cloners in the last 14 days | int |
| clones_by_period | Clones and unique cloners for each day or week, oldest first | List[TrafficCount] |
| referrers | Top 10 sites that sent visitors in the last 14 days, with the views and unique visitors from each | List[TrafficReferrer] |
| popular_content | Top 10 most viewed pages of the repository in the last 14 days, with the views, unique visitors and github.com link of each | List[PopularContent] |

### Possible use case
<!-- MANUAL: use_case -->
**Traffic History**: Run the block on a daily schedule and append yesterday's views and clones to a Google Sheet, so you keep more than the 14 days GitHub holds.

**Weekly Report**: Every Monday, post last week's views, unique visitors, clones and top referrers to Slack.

**Launch Check**: After a release or a blog post, see whether views and clones went up and which sites sent the visitors.
<!-- END MANUAL -->

---
