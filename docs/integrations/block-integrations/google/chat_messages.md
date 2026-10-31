# Google Chat Messages
<!-- MANUAL: file_description -->
Blocks for reading and sending messages in one Google Chat conversation. List Messages reads with the `chat.messages.readonly` scope, which Google classes as restricted, so a production app needs Google's restricted-scope verification. Send Message posts as the user with `chat.messages.create` (sensitive), and also reads the space's details with `chat.spaces.readonly` (sensitive) to check a thread reply is possible before sending.

Setup: the Google Chat API must be enabled in the Google Cloud project behind AutoGPT's Google sign-in. Sending is a create call, so it also needs a Google Chat app (name, avatar and description) configured in that project ([Configure the Google Chat API](https://developers.google.com/workspace/chat/configure-chat-api)). Google Chat shows that app's name next to the user's name on messages sent this way.
<!-- END MANUAL -->

## Google Chat List Messages

### What it is
Read the messages in a Google Chat space, group chat or direct message, optionally only one thread or a time range. Returns each message's text, sender, time and thread.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Chat API `spaces.messages.list` endpoint for one conversation, newest first unless you turn that off. The thread and time range become a Chat API filter, and times without a time zone count as UTC. Each message has its text, sender, time, thread and attachments; system messages, such as someone joining, aren't included.

The conversation can be a space ID or a Google Chat link, and the thread a thread ID or its full `spaces/.../threads/...` name; a thread from a different conversation stops with an input error. Google can leave out a sender's name and email, for example for someone who has left the space.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| space | The conversation to read: its space ID (spaces/...) or a Google Chat link | str | Yes |
| thread_id | Only messages in this thread (spaces/.../threads/...) | str | No |
| created_after | Only messages sent after this time | str (date-time) | No |
| created_before | Only messages sent before this time | str (date-time) | No |
| newest_first | Return the newest messages first | bool | No |
| max_results | Maximum number of messages to return | int | No |
| page_token | Page token from a previous call, to get the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| messages | Messages, newest first unless that is turned off | List[ChatMessage] |
| message | Each message | ChatMessage |
| next_page_token | Token for the next page, when there are more messages | str |

### Possible use case
<!-- MANUAL: use_case -->
**Discussion Summaries**: Summarise yesterday's discussion in a project space for people who missed it.

**Thread Context**: Read a whole thread before drafting a reply to it with Google Chat Send Message.

**Chat Archiving**: Copy a month of messages from a space into a document or spreadsheet for record keeping.
<!-- END MANUAL -->

---

## Google Chat Send Message

### What it is
Send a Google Chat message as the user to a space, group chat or direct message, or reply in a thread of a named space. Returns the sent message.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Chat API `spaces.messages.create` endpoint to post the text as the user, read as Markdown unless you turn that off. Empty text and text over Google's 32,000-byte limit stop with an input error before anything is sent. The block is marked as an irreversible action, because a sent message reaches other people.

With a `thread_id`, it first checks the space: the Chat API only posts thread replies in named spaces that have threads, so for a direct message, a group chat or an unthreaded space the block stops with an error before sending anything, rather than posting the reply to the whole conversation. Replies use `REPLY_MESSAGE_OR_FAIL`, so a thread that doesn't exist also fails instead of starting a new one.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| space | Where to send it: a space ID (spaces/...) or a Google Chat link. Get a direct message's ID from Google Chat Start Direct Message. | str | Yes |
| text | The message, up to 32,000 bytes. Markdown works for bold, italics, code, links and lists. Mention someone with `<chat-user data-email="name@example.com">`. | str | Yes |
| thread_id | Reply in this thread (spaces/.../threads/...) instead of starting a new one. The Chat API only takes thread replies in named spaces. | str | No |
| markdown | Read the text as Markdown. Turn off to use Google Chat's own formatting instead: `*bold*`, `_italic_`, `<users/123>` mentions. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message | The sent message, with its ID and thread for follow-up replies | ChatMessage |

### Possible use case
<!-- MANUAL: use_case -->
**Status Updates**: Post a daily status update to a team space.

**Incident Replies**: Reply in the thread of an incident alert with the latest findings.

**Meeting Follow-ups**: Send a follow-up to a colleague after a meeting, using Google Chat Start Direct Message to get the conversation.
<!-- END MANUAL -->

---
