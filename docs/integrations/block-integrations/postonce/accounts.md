# PostOnce Accounts
<!-- MANUAL: file_description -->
[PostOnce](https://postonce.to) publishes and schedules posts to LinkedIn, X, Threads, Bluesky, Facebook Pages, Instagram, TikTok, YouTube, Pinterest and more through each network's official API. These blocks call the PostOnce public API (`https://postonce.to/api/public/v1`) with your own API key, created under Developers in the PostOnce dashboard and sent as `Authorization: Bearer <key>`. Social accounts are connected in PostOnce, not in AutoGPT. PostOnce bills your PostOnce plan; AutoGPT adds no charge. API reference: [docs.postonce.to](https://docs.postonce.to).
<!-- END MANUAL -->

## PostOnce List Accounts

### What it is
List the social accounts connected to PostOnce, with the IDs Create Post needs.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `GET /accounts`, which returns every social account connected to your PostOnce workspace, including ones that need reconnecting. With `active_only` on (the default), accounts whose status is not `active` are left out. Set `platform` to keep one network only. Each account carries its `capabilities` and `media_requirements`, such as whether the network needs media on every post.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| platform | Only list accounts on this platform | "bluesky" \| "facebook" \| "farcaster" \| "google_business" \| "instagram" \| "linkedin" \| "nostr" \| "pinterest" \| "reddit" \| "snapchat" \| "telegram" \| "threads" \| "tiktok" \| "twitter" \| "vimeo" \| "youtube" | No |
| active_only | Leave out accounts that need reconnecting | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| accounts | Connected accounts: id, platform, username, status, capabilities, media_requirements | List[Dict[str, Any]] |
| account_ids | IDs of the listed accounts, ready for Create Post | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Pick where to post**: List the active accounts at the start of a workflow and pass `account_ids` straight into PostOnce Create Post.

**Reconnect reminders**: Turn off `active_only` and send yourself a message for every account whose status is `auth_required`.
<!-- END MANUAL -->

---
