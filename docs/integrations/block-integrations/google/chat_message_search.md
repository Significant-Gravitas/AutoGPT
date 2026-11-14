# Google Chat Message Search
<!-- MANUAL: file_description -->
A block that searches Google Chat messages across all the user's conversations, using the Chat API's message search (generally available since July 2026). It reads with `chat.messages.readonly`, which Google classes as restricted, so a production app needs Google's restricted-scope verification. It also asks for `chat.spaces.readonly` (for the conversation-type and space-name filters) and `chat.users.readstate.readonly` (for the unread filter); both are sensitive.

Setup: the Google Chat API must be enabled in the Google Cloud project behind AutoGPT's Google sign-in. Read-only calls like this one don't need a configured Chat app ([Configure the Google Chat API](https://developers.google.com/workspace/chat/configure-chat-api)).
<!-- END MANUAL -->

## Google Chat Search Messages

### What it is
Search Google Chat messages across every conversation the user is in, by keywords, sender, conversation, time, unread status, mentions, links or attachments. Returns each message's text, sender, time, conversation and thread.

### How it works
<!-- MANUAL: how_it_works -->
Turns the filters you set into one Chat API search query and calls the `spaces.messages.search` endpoint across every conversation (`spaces/-`). Keywords are passed on as typed, so quoted phrases work, and results come newest first. The space-name filter only looks in the 5 spaces whose names match best, and the block stops with an input error if you set no keyword or filter.

Google leaves some messages out of search: messages from Chat apps, direct messages with Chat apps, messages from blocked people, messages in muted spaces and private messages. Use Google Chat List Messages to read everything in one conversation.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| keywords | Words the messages must contain. Put a phrase in double quotes, e.g. "launch checklist". | str | No |
| unread_only | Only messages the user hasn't read | bool | No |
| mentions_me | Only messages that @mention the user | bool | No |
| space | Only messages in this conversation (space ID or Google Chat link) | str | No |
| sender | Only messages from this person (email address or users/...) | str | No |
| created_after | Only messages sent at or after this time | str (date-time) | No |
| created_before | Only messages sent before this time | str (date-time) | No |
| space_type | Only messages in conversations of this kind | "any" \| "space" \| "group_chat" \| "direct_message" | No |
| space_name_contains | Only messages in spaces whose name contains this text. Google searches the 5 best-matching spaces. | str | No |
| has_link | Only messages that contain a link | bool | No |
| has_attachment | Only messages with an attachment | bool | No |
| max_results | Maximum number of messages to return | int | No |
| page_token | Page token from a previous search, to get the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| messages | Matching messages, newest first | List[ChatMessage] |
| message | Each matching message | ChatMessage |
| next_page_token | Token for the next page, when there are more results | str |

### Possible use case
<!-- MANUAL: use_case -->
**Unread Mentions**: Find unread messages that mention the user and list what needs a reply.

**Topic Research**: Find everything a colleague said about a renewal this month.

**Link Digest**: Collect messages with links shared in a project space for a weekly digest.
<!-- END MANUAL -->

---
