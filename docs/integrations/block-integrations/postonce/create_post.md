# PostOnce Create Post
<!-- MANUAL: file_description -->
[PostOnce](https://postonce.to) publishes and schedules posts to LinkedIn, X, Threads, Bluesky, Facebook Pages, Instagram, TikTok, YouTube, Pinterest and more through each network's official API. These blocks call the PostOnce public API (`https://postonce.to/api/public/v1`) with your own API key, created under Developers in the PostOnce dashboard and sent as `Authorization: Bearer <key>`. Social accounts are connected in PostOnce, not in AutoGPT. PostOnce bills your PostOnce plan; AutoGPT adds no charge. API reference: [docs.postonce.to](https://docs.postonce.to).
<!-- END MANUAL -->

## PostOnce Create Post

### What it is
Publish or schedule one post to several social accounts with PostOnce, with optional media and per-account text.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one `POST /posts` request with a target for each account ID, the post text, any media URLs and, if set, `publish_at`. Without `publish_at` PostOnce publishes right away; with it the post is scheduled. `content_overrides` and `platform_options` (for example a YouTube title) are keyed by account ID and only allowed for accounts in `account_ids`. Media URLs must be public HTTP(S) links PostOnce can download. The request is sent once and not retried; set `idempotency_key` so a rerun with the same input returns the original post instead of posting twice. Delivery happens after the block returns: `status` and each target's `status` start as queued or scheduled, and PostOnce Get Post reports the outcome and published links. A rejected request (unknown or disconnected account, invalid media) fails the block with PostOnce's error code and message.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| account_ids | IDs of the connected accounts to post to, from PostOnce List Accounts | List[str] | Yes |
| content | Post text | str | No |
| media_urls | Public HTTP(S) URLs of images or videos to attach | List[str] | No |
| publish_at | When to publish (UTC if no time zone is given). Leave empty to publish now. | str (date-time) | No |
| content_overrides | Different text for some accounts, keyed by account ID | Dict[str, str] | No |
| platform_options | Platform settings keyed by account ID, e.g. {"<youtube account>": {"title": "Launch video"}} | Dict[str, Dict[str, Any]] | No |
| external_id | Your own reference for this post | str | No |
| idempotency_key | Reusing a key with the same input returns the original post instead of posting twice | str | No |

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
**Cross-post a launch**: Publish one announcement with an image to LinkedIn, X, Threads and Bluesky, with a shorter version for X.

**Content calendar**: Generate a week of posts with an AI block and schedule each one with its own `publish_at`.

**Video repurposing**: Send one rendered video URL to TikTok, Instagram Reels and YouTube Shorts with a title for YouTube.
<!-- END MANUAL -->

---
