# Twitter Hide
<!-- MANUAL: file_description -->
Blocks for hiding and unhiding tweet replies on Twitter/X.
<!-- END MANUAL -->

## Twitter Hide Reply

### What it is
This block hides a reply to a tweet.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to hide a reply to one of your tweets. Hidden replies are not deleted—they're moved behind a "View hidden replies" option that viewers can click to see. Only the original tweet author can hide replies.

The block authenticates using OAuth 2.0 and sends a PUT request to change the reply's hidden status. This is useful for managing conversation threads and reducing visibility of off-topic or inappropriate replies.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| tweet_id | ID of the tweet reply to hide | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the operation was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Conversation Moderation**: Hide off-topic or spam replies to keep your tweet threads focused and readable.

**Brand Protection**: Hide inappropriate or offensive replies to maintain a professional appearance on brand accounts.

**Community Management**: Moderate discussions by hiding replies that violate community guidelines without deleting them entirely.
<!-- END MANUAL -->

---

## Twitter Unhide Reply

### What it is
This block unhides a reply to a tweet.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to restore visibility of a previously hidden reply. The reply will appear normally in the conversation thread again.

The block authenticates using OAuth 2.0 and sends a PUT request to change the reply's hidden status back to visible. Only the original tweet author can unhide replies they previously hid.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| tweet_id | ID of the tweet reply to unhide | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the operation was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Moderation Review**: Restore replies after reviewing them and determining they were hidden incorrectly.

**Context Restoration**: Unhide replies that provide important context that was initially overlooked.

**User Appeals**: Restore hidden replies after a user explains their intent or edits problematic content.
<!-- END MANUAL -->

---
