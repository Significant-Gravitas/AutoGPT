# Google Chat Spaces
<!-- MANUAL: file_description -->
Blocks for finding Google Chat conversations: list the spaces, group chats and direct messages the user is in, search named spaces by name, and find group chats by who is in them. They return conversation IDs (`spaces/...`) that the Google Chat message blocks take. They only read, with the `chat.spaces.readonly` scope; Find Group Chats also asks for `chat.memberships.readonly`. Google classes both as sensitive.

Setup: the Google Chat API must be enabled in the Google Cloud project behind AutoGPT's Google sign-in, and Google built the Chat API for Google Workspace accounts. Read-only calls like these don't need a configured Chat app ([Configure the Google Chat API](https://developers.google.com/workspace/chat/configure-chat-api)).
<!-- END MANUAL -->

## Google Chat Find Group Chats

### What it is
Find Google Chat group chats whose members are exactly the user plus the people you list, by email address or user ID, and get their IDs for reading or sending messages.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Chat API `spaces.findGroupChats` endpoint with the people you list, as `users/<email or ID>`, and asks for full space details. It returns group chats whose joined members are exactly the user plus those people: a chat with anyone else in it is left out, while Chat apps in the chat don't count.

Duplicate and blank entries are dropped before the call. The block stops with an input error if no one is left or the list has more than 49 people, the most Google accepts. Use `next_page_token` to get more results.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| people | Email addresses or Chat user IDs (users/...) of everyone else in the group chat, not including you. Up to 49. | List[str] | Yes |
| max_results | Maximum number of group chats to return | int | No |
| page_token | Page token from a previous call, to get the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| spaces | Matching group chats | List[ChatSpace] |
| space | Each matching group chat | ChatSpace |
| next_page_token | Token for the next page, when there are more results | str |

### Possible use case
<!-- MANUAL: use_case -->
**Team Chat Lookup**: Find the group chat with two teammates and summarise its latest messages with Google Chat List Messages.

**Avoid Duplicate Chats**: Check whether a group chat with a set of people already exists before starting a new conversation.

**Project Follow-ups**: Find the group chat of a project's core members and post a status update there with Google Chat Send Message.
<!-- END MANUAL -->

---

## Google Chat List Spaces

### What it is
List the Google Chat conversations the user is in (named spaces, group chats and direct messages) with their IDs, names, types and member counts.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Chat API `spaces.list` endpoint, optionally filtered to one kind of conversation: named spaces, group chats or direct messages. Each result has the conversation's ID, name (named spaces only), type, link, member count and last activity time.

Group chats and direct messages only appear once someone has sent a message in them. Use `next_page_token` to page through long lists; each page holds up to 1,000 conversations.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| space_type | Only conversations of this kind | "any" \| "space" \| "group_chat" \| "direct_message" | No |
| max_results | Maximum number of conversations to return | int | No |
| page_token | Page token from a previous call, to get the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| spaces | Conversations the user is in | List[ChatSpace] |
| space | Each conversation | ChatSpace |
| next_page_token | Token for the next page, when there are more results | str |

### Possible use case
<!-- MANUAL: use_case -->
**Daily Digest**: List every space the user belongs to and summarise the new activity in each one.

**Space ID Lookup**: Find a conversation's ID before reading or sending messages with the other Google Chat blocks.

**Space Inventory**: Export the user's spaces with their member counts and last activity to a spreadsheet to spot inactive ones.
<!-- END MANUAL -->

---

## Google Chat Search Spaces

### What it is
Find named Google Chat spaces the user is in by words in the space name, and get their IDs. Direct messages and group chats have no name: use Find Direct Message or Find Group Chats.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Chat API `spaces.search` endpoint without administrator rights, so it only searches named spaces the user has joined. Each word you give must match the start of a word in the space name, in any order, so `launch plan` finds "Q4 Launch Planning".

Google returns up to 100 matches and no further pages. Direct messages and group chats have no name, so they never match: use Google Chat Find Direct Message or Find Group Chats for those. An empty search stops with an input error.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name_query | Words from the space's name. Each word matches the start of a word in the name, in any order: 'launch plan' finds 'Q4 Launch Planning'. | str | Yes |
| max_results | Maximum number of spaces to return | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| spaces | Matching spaces | List[ChatSpace] |
| space | Each matching space | ChatSpace |

### Possible use case
<!-- MANUAL: use_case -->
**Post by Space Name**: Find the "Launch planning" space by name and post an update in it with Google Chat Send Message.

**Plain-Language Targets**: Let a user name a space in their own words instead of pasting its ID.

**Space Discovery**: Find every space with "support" in its name before collecting their recent messages.
<!-- END MANUAL -->

---
