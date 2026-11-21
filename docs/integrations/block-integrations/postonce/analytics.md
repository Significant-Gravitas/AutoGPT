# PostOnce Analytics
<!-- MANUAL: file_description -->
[PostOnce](https://postonce.to) publishes and schedules posts to LinkedIn, X, Threads, Bluesky, Facebook Pages, Instagram, TikTok, YouTube, Pinterest and more through each network's official API. These blocks call the PostOnce public API (`https://postonce.to/api/public/v1`) with your own API key, created under Developers in the PostOnce dashboard and sent as `Authorization: Bearer <key>`. Social accounts are connected in PostOnce, not in AutoGPT. PostOnce bills your PostOnce plan; AutoGPT adds no charge. API reference: [docs.postonce.to](https://docs.postonce.to).
<!-- END MANUAL -->

## PostOnce Get Post Analytics

### What it is
Get views, likes, comments, shares and saves for a published PostOnce post, per platform and in total.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `GET /posts/{id}/analytics`, which returns current views, likes, comments, shares and saves for each copy published on TikTok, Instagram, YouTube or Facebook. Other networks and copies not yet published are left out. Numbers are cached by PostOnce for up to an hour; `last_updated` on each target says when they were read, and `metrics_status` says when an account must be reconnected to read them. The totals add up every copy, counting a metric a network does not report as 0.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post_id | PostOnce post ID, from the post_id output of Create Post | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| targets | One entry per published copy on TikTok, Instagram, YouTube or Facebook: platform, url, views, likes, comments, shares, saves, engagement_rate, metrics_status, last_updated | List[Dict[str, Any]] |
| views | Views across all copies | int |
| likes | Likes across all copies | int |
| comments | Comments across all copies | int |
| shares | Shares across all copies | int |
| saves | Saves across all copies | int |

### Possible use case
<!-- MANUAL: use_case -->
**Performance report**: A day after posting, read each post's totals and write them to a sheet.

**Repost winners**: Compare views across platforms and reschedule the best performer elsewhere.
<!-- END MANUAL -->

---
