# Discord Bot Blocks
<!-- MANUAL: file_description -->
Blocks for interacting with Discord using bot tokens, including sending messages, managing threads, and reading channel data.
<!-- END MANUAL -->

## Create Discord Thread

### What it is
Creates a new thread in a Discord channel.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Discord API with a bot token to create a new thread in a specified channel. Threads can be public or private (private requires Boost Level 2+).

Configure auto-archive duration and optionally send an initial message when the thread is created.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| channel_name | Channel ID or channel name to create the thread in | str | Yes |
| server_name | Server name (only needed if using channel name) | str | No |
| thread_name | The name of the thread to create | str | Yes |
| is_private | Whether to create a private thread (requires Boost Level 2+) or public thread | bool | No |
| auto_archive_duration | Duration before the thread is automatically archived | "60" \| "1440" \| "4320" \| "10080" | No |
| message_content | Optional initial message to send in the thread | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| status | Operation status | str |
| thread_id | ID of the created thread | str |
| thread_name | Name of the created thread | str |

### Possible use case
<!-- MANUAL: use_case -->
**Support Tickets**: Create threads for individual support conversations to keep channels organized.

**Discussion Topics**: Automatically create threads for new topics or announcements.

**Project Channels**: Spin up discussion threads for specific tasks or features.
<!-- END MANUAL -->

---

## Discord Channel Info

### What it is
Resolves Discord channel names to IDs and vice versa.

### How it works
<!-- MANUAL: how_it_works -->
This block resolves Discord channel identifiers, converting between channel names and IDs. It queries the Discord API to find the channel and returns comprehensive information including server details.

Useful for workflows that receive channel names but need IDs for other Discord operations.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| channel_identifier | Channel name or channel ID to look up | str | Yes |
| server_name | Server name (optional, helps narrow down search) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| channel_id | The channel's ID | str |
| channel_name | The channel's name | str |
| server_id | The server's ID | str |
| server_name | The server's name | str |
| channel_type | Type of channel (text, voice, etc) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Dynamic Routing**: Look up channel IDs to route messages to user-specified channels by name.

**Validation**: Verify channel existence before attempting to send messages.

**Workflow Setup**: Get channel details during workflow configuration.
<!-- END MANUAL -->

---

## Discord User Info

### What it is
Gets information about a Discord user by their ID.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves information about a Discord user by their ID. It queries the Discord API and returns profile details including username, display name, avatar, and account creation date.

The user must be visible to your bot (share a server with your bot).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| user_id | The Discord user ID to get information about | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| user_id | The user's ID (passed through for chaining) | str |
| username | The user's username | str |
| display_name | The user's display name | str |
| discriminator | The user's discriminator (if applicable) | str |
| avatar_url | URL to the user's avatar | str |
| is_bot | Whether the user is a bot | bool |
| created_at | When the account was created | str |

### Possible use case
<!-- MANUAL: use_case -->
**User Profiling**: Get user details to personalize responses or create user profiles.

**Mention Resolution**: Look up user information when processing mentions in messages.

**Activity Logging**: Retrieve user details for logging or analytics purposes.
<!-- END MANUAL -->

---

## Read Discord Messages

### What it is
Reads messages from a Discord channel using a bot token.

### How it works
<!-- MANUAL: how_it_works -->
The block uses a Discord bot to log into a server and listen for new messages. When a message is received, it extracts the content, channel name, and username of the sender. If the message contains a text file attachment, the block also retrieves and includes the file's content.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_content | The content of the message received | str |
| message_id | The ID of the message | str |
| channel_id | The ID of the channel | str |
| channel_name | The name of the channel the message was received from | str |
| user_id | The ID of the user who sent the message | str |
| username | The username of the user who sent the message | str |

### Possible use case
<!-- MANUAL: use_case -->
This block could be used to monitor a Discord channel for support requests. When a user posts a message, the block captures it, allowing another part of the system to process and respond to the request.
<!-- END MANUAL -->

---

## Reply To Discord Message

### What it is
Replies to a specific Discord message.

### How it works
<!-- MANUAL: how_it_works -->
This block sends a reply to a specific Discord message, creating a threaded reply that references the original message. Optionally mention the original author to notify them.

The reply appears linked to the original message in Discord's UI, maintaining conversation context.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| channel_id | The channel ID where the message to reply to is located | str | Yes |
| message_id | The ID of the message to reply to | str | Yes |
| reply_content | The content of the reply | str | Yes |
| mention_author | Whether to mention the original message author | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| status | Operation status | str |
| reply_id | ID of the reply message | str |

### Possible use case
<!-- MANUAL: use_case -->
**Conversation Bots**: Reply to user questions maintaining conversation context.

**Support Responses**: Respond to support requests by replying to the original message.

**Interactive Commands**: Reply to command messages with results or confirmations.
<!-- END MANUAL -->

---

## Send Discord DM

### What it is
Sends a direct message to a Discord user using their user ID.

### How it works
<!-- MANUAL: how_it_works -->
This block sends a direct message to a Discord user. It opens a DM channel with the user (if not already open) and sends the message. The user must allow DMs from server members or share a server with your bot.

Returns the message ID of the sent DM for tracking purposes.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| user_id | The Discord user ID to send the DM to (e.g., '123456789012345678') | str | Yes |
| message_content | The content of the direct message to send | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| status | The status of the operation | str |
| message_id | The ID of the sent message | str |

### Possible use case
<!-- MANUAL: use_case -->
**Private Notifications**: Send private alerts or notifications to specific users.

**Welcome Messages**: DM new server members with welcome information.

**Verification Systems**: Send verification codes or instructions via DM.
<!-- END MANUAL -->

---

## Send Discord Embed

### What it is
Sends a rich embed message to a Discord channel.

### How it works
<!-- MANUAL: how_it_works -->
This block sends a rich embed message to a Discord channel. Embeds support formatted content with titles, descriptions, colors, images, thumbnails, author sections, footers, and structured fields.

Configure the embed's appearance with colors, images, and multiple fields for organized information display.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| channel_identifier | Channel ID or channel name to send the embed to | str | Yes |
| server_name | Server name (only needed if using channel name) | str | No |
| title | The title of the embed | str | No |
| description | The main content/description of the embed | str | No |
| color | Embed color as integer (e.g., 0x00ff00 for green) | int | No |
| thumbnail_url | URL for the thumbnail image | str | No |
| image_url | URL for the main embed image | str | No |
| author_name | Author name to display | str | No |
| footer_text | Footer text | str | No |
| fields | List of field dictionaries with 'name', 'value', and optional 'inline' keys | List[Dict[str, Any]] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| status | Operation status | str |
| message_id | ID of the sent embed message | str |

### Possible use case
<!-- MANUAL: use_case -->
**Status Updates**: Send formatted status updates with colors and structured information.

**Data Displays**: Present data in organized embed fields for easy reading.

**Announcements**: Create visually appealing announcements with images and branding.
<!-- END MANUAL -->

---

## Send Discord File

### What it is
Sends a file attachment to a Discord channel.

### How it works
<!-- MANUAL: how_it_works -->
This block uploads and sends a file attachment to a Discord channel. It supports various file types including images, documents, and other media. Files can be provided as URLs, data URIs, or local paths.

Optionally include a message along with the file attachment.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| channel_identifier | Channel ID or channel name to send the file to | str | Yes |
| server_name | Server name (only needed if using channel name) | str | No |
| file | The file to send (URL, data URI, or local path). Supports images, videos, documents, etc. | str (file) | Yes |
| filename | Name of the file when sent (e.g., 'report.pdf', 'image.png') | str | No |
| message_content | Optional message to send with the file | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| status | Operation status | str |
| message_id | ID of the sent message | str |

### Possible use case
<!-- MANUAL: use_case -->
**Report Sharing**: Send generated reports or documents to Discord channels.

**Image Posting**: Share images from workflows or external sources.

**Backup Distribution**: Share backup files or exports with team channels.
<!-- END MANUAL -->

---

## Send Discord Message

### What it is
Sends a message to a Discord channel using a bot token.

### How it works
<!-- MANUAL: how_it_works -->
The block uses a Discord bot to log into a server, locate the specified channel, and send the provided message. If the message is longer than Discord's character limit, it automatically splits the message into smaller chunks and sends them sequentially.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| message_content | The content of the message to send | str | Yes |
| channel_name | Channel ID or channel name to send the message to | str | Yes |
| server_name | Server name (only needed if using channel name) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| status | The status of the operation (e.g., 'Message sent', 'Error') | str |
| message_id | The ID of the sent message | str |
| channel_id | The ID of the channel where the message was sent | str |

### Possible use case
<!-- MANUAL: use_case -->
This block could be used as part of an automated notification system. For example, it could send alerts to a Discord channel when certain events occur in another system, such as when a new user signs up or when a critical error is detected.
<!-- END MANUAL -->

---
