# Slack Blocks
<!-- MANUAL: file_description -->
Blocks for sending messages to Slack channels, direct messages, and threads using a Slack Bot Token.
<!-- END MANUAL -->

## Send Slack Message

### What it is
Send a text message to any Slack channel, DM, or thread. Required bot token scope: chat:write.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Slack `chat.postMessage` API with your bot token. The token must have the `chat:write` scope. Optionally uses `chat:write.customize` to set a custom username or icon emoji. Returns the message timestamp (`ts`), which can be passed back as `thread_ts` to reply in the same thread.

If the Slack API returns a non-`ok` response (e.g. `invalid_auth`, `channel_not_found`, `not_in_channel`, or missing scopes), the block raises a user-facing error with the Slack error code so you can diagnose and fix the issue. Network failures and unexpected exceptions are also surfaced as errors through the block's `error` output.
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
**Daily Summaries**: Post a recap of completed tasks or pipeline results to a `#updates` channel at the end of each run.

**Anomaly Alerts**: Send an immediate notification to a `#alerts` channel when a monitoring workflow detects an error or threshold breach.

**Threaded Follow-ups**: Chain two Send Slack Message blocks — pass the `ts` output of the first into the `thread_ts` input of the second to keep responses organized in a single thread.
<!-- END MANUAL -->

---
