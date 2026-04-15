# Ayrshare Post To X
<!-- MANUAL: file_description -->
Blocks for posting tweets and threads to X (Twitter) using the Ayrshare social media management API.
<!-- END MANUAL -->

## Post To X

### What it is
Post to X / Twitter using Ayrshare

### How it works
<!-- MANUAL: how_it_works -->
This block uses Ayrshare's API to publish content to X (formerly Twitter). It supports standard tweets (280 characters, or 25,000 for Premium users), threads, polls, quote tweets, and replies, with up to 4 media attachments including video with subtitles.

The block authenticates through Ayrshare and handles X-specific features like automatic thread breaking using double newlines, thread numbering, per-post media attachments, and long-form video uploads (with approval). Poll options and duration can be configured for engagement posts.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post | The post text (max 280 chars, up to 25,000 for Premium users). Use @handle to mention users. Use \n\n for thread breaks. | str | Yes |
| media_urls | Optional list of media URLs. X supports up to 4 images or videos per tweet. Auto-preview links unless media is included. | List[str] | No |
| is_video | Whether the media is a video | bool | No |
| schedule_date | UTC datetime for scheduling (YYYY-MM-DDThh:mm:ssZ) | str (date-time) | No |
| disable_comments | Whether to disable comments | bool | No |
| shorten_links | Whether to shorten links | bool | No |
| unsplash | Unsplash image configuration | str | No |
| requires_approval | Whether to enable approval workflow | bool | No |
| random_post | Whether to generate random post text | bool | No |
| random_media_url | Whether to generate random media | bool | No |
| notes | Additional notes for the post | str | No |
| reply_to_id | ID of the tweet to reply to | str | No |
| quote_tweet_id | ID of the tweet to quote (low-level Tweet ID) | str | No |
| poll_options | Poll options (2-4 choices) | List[str] | No |
| poll_duration | Poll duration in minutes (1-10080) | int | No |
| alt_text | Alt text for each image (max 1,000 chars each, not supported for videos) | List[str] | No |
| is_thread | Whether to automatically break post into thread based on line breaks | bool | No |
| thread_number | Add thread numbers (1/n format) to each thread post | bool | No |
| thread_media_urls | Media URLs for thread posts (one per thread, use 'null' to skip) | List[str] | No |
| long_post | Force long form post (requires Premium X account) | bool | No |
| long_video | Enable long video upload (requires approval and Business/Enterprise plan) | bool | No |
| subtitle_url | URL to SRT subtitle file for videos (must be HTTPS and end in .srt) | str | No |
| subtitle_language | Language code for subtitles (default: 'en') | str | No |
| subtitle_name | Name of caption track (max 150 chars, default: 'English') | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| post_result | The result of the post | PostResponse |
| post | The result of the post | PostIds |

### Possible use case
<!-- MANUAL: use_case -->
**Thread Publishing**: Automatically format and publish long-form content as numbered thread sequences.

**Engagement Polls**: Create polls to gather audience feedback or drive interaction with scheduled posting.

**Reply Automation**: Build workflows that automatically respond to mentions or engage in conversations.
<!-- END MANUAL -->

---
