# Slack Blocks
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Send Slack Message

### What it is
Send a text message to any Slack channel, DM, or thread. Required bot token scope: chat:write.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| channel | Channel ID (e.g. C1234567890) or name (e.g. #general). For DMs use the user's member ID (e.g. U1234567890). | str | Yes |
| text | Message text. Supports Slack mrkdwn: *bold*, _italic_, `code`, <https://example.com\|link>. | str | Yes |
| thread_ts | Timestamp of the parent message to reply in a thread. Use the 'ts' output from a previous Send Slack Message block. | str | No |
| username | Custom display name for the bot in this message. Requires chat:write.customize scope. | str | No |
| icon_emoji | Emoji to use as the bot's icon (e.g. :robot_face:). Requires chat:write.customize scope. | str | No |
| unfurl_links | Automatically expand URLs into rich previews. | bool | No |
| mrkdwn | Enable Slack markdown formatting in the message text. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ts | Message timestamp — Slack's unique message ID. Use as 'thread_ts' to reply in a thread. | str |
| channel | The channel ID where the message was posted. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
