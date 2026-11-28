# PostOnce Posts
<!-- MANUAL: file_description -->
[PostOnce](https://postonce.to) publishes and schedules posts to LinkedIn, X, Threads, Bluesky, Facebook Pages, Instagram, TikTok, YouTube, Pinterest and more through each network's official API. These blocks call the PostOnce public API (`https://postonce.to/api/public/v1`) with your own API key, created under Developers in the PostOnce dashboard and sent as `Authorization: Bearer <key>`. Social accounts are connected in PostOnce, not in AutoGPT. PostOnce bills your PostOnce plan; AutoGPT adds no charge. API reference: [docs.postonce.to](https://docs.postonce.to).
<!-- END MANUAL -->

## PostOnce Cancel Post

### What it is
Cancel a scheduled or still-pending PostOnce post. Copies that already went out stay published.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `POST /posts/{id}/cancel`. PostOnce stops every target that has not gone out yet and marks it cancelled; copies already published stay on the networks, and the post record is kept. The block returns the updated post.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post_id | PostOnce post ID, from the post_id output of Create Post | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| post_id | PostOnce post ID | str |
| status | queued, scheduled, processing, partial, published, failed or cancelled | str |
| targets | One entry per account: account_id, platform, username, status, error, platform_post_id, platform_post_url, scheduled_time | List[Dict[str, Any]] |
| post_urls | Links to the published posts; empty until a target publishes | List[str] |
| post | The full post object PostOnce returned | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Approval gate**: Schedule a post, wait for a reviewer, and cancel it if they reject it.

**Stop outdated posts**: Cancel scheduled posts about an offer once it sells out.
<!-- END MANUAL -->

---

## PostOnce Get Post

### What it is
Get a PostOnce post's status and, for each account, its delivery state, error and published link.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `GET /posts/{id}` and returns the post's overall `status` and one entry per target with its delivery `status`, `error`, and `platform_post_url` once published. `post_urls` collects the published links. A post is `partial` when some targets published and others failed.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post_id | PostOnce post ID, from the post_id output of Create Post | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| post_id | PostOnce post ID | str |
| status | queued, scheduled, processing, partial, published, failed or cancelled | str |
| targets | One entry per account: account_id, platform, username, status, error, platform_post_id, platform_post_url, scheduled_time | List[Dict[str, Any]] |
| post_urls | Links to the published posts; empty until a target publishes | List[str] |
| post | The full post object PostOnce returned | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Confirm delivery**: Check a post a few minutes after creating it and alert someone when a target failed, with the error.

**Share the links**: Send the published links to a Slack channel or write them to a spreadsheet.
<!-- END MANUAL -->

---
