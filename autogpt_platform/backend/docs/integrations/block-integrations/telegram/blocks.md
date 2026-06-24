# Telegram Blocks
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Delete Telegram Message

### What it is
Delete a message from a Telegram chat. Bots can delete their own messages and incoming messages in private chats at any time.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| chat_id | The chat ID containing the message | int | Yes |
| message_id | The ID of the message to delete | int | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| status | Status of the operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Edit Telegram Message

### What it is
Edit the text of an existing message sent by the bot.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| chat_id | The chat ID containing the message | int | Yes |
| message_id | The ID of the message to edit | int | Yes |
| text | New text for the message (max 4096 characters) | str | Yes |
| parse_mode | Message formatting mode | "none" \| "Markdown" \| "MarkdownV2" \| "HTML" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_id | The ID of the edited message | int |
| status | Status of the operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Get Telegram File

### What it is
Download a file from Telegram using its file_id. Use this to process photos, voice messages, or documents received.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| file_id | The Telegram file_id to download. Get this from trigger outputs (photo_file_id, voice_file_id, etc.) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| file | The downloaded file (workspace:// reference or data URI) | str (file) |
| status | Status of the operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Reply To Telegram Message

### What it is
Reply to a specific message in a Telegram chat.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| chat_id | The chat ID where the message is | int | Yes |
| reply_to_message_id | The message ID to reply to | int | Yes |
| text | The reply text | str | Yes |
| parse_mode | Message formatting mode | "none" \| "Markdown" \| "MarkdownV2" \| "HTML" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_id | The ID of the reply message | int |
| status | Status of the operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Send Telegram Audio

### What it is
Send an audio file to a Telegram chat. The file is displayed in the music player. For voice messages, use the Send Voice block instead.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| chat_id | The chat ID to send the audio to | int | Yes |
| audio | Audio file to send (MP3 or M4A format). Can be URL, data URI, or workspace:// reference. | str (file) | Yes |
| caption | Caption for the audio file | str | No |
| title | Track title | str | No |
| performer | Track performer/artist | str | No |
| duration | Duration in seconds | int | No |
| reply_to_message_id | Message ID to reply to | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_id | The ID of the sent message | int |
| status | Status of the operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Send Telegram Document

### What it is
Send a document (any file type) to a Telegram chat.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| chat_id | The chat ID to send the document to | int | Yes |
| document | Document to send (any file type). Can be URL, data URI, or workspace:// reference. | str (file) | Yes |
| filename | Filename shown to the recipient. If empty, the original filename is used (may be a random ID for uploaded files). | str | No |
| caption | Caption for the document | str | No |
| parse_mode | Caption formatting mode | "none" \| "Markdown" \| "MarkdownV2" \| "HTML" | No |
| reply_to_message_id | Message ID to reply to | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_id | The ID of the sent message | int |
| status | Status of the operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Send Telegram Message

### What it is
Send a text message to a Telegram chat.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| chat_id | The chat ID to send the message to. Get this from the trigger block's chat_id output. | int | Yes |
| text | The text message to send (max 4096 characters) | str | Yes |
| parse_mode | Message formatting mode (Markdown, HTML, or none) | "none" \| "Markdown" \| "MarkdownV2" \| "HTML" | No |
| reply_to_message_id | Message ID to reply to | int | No |
| disable_notification | Send message silently (no notification sound) | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_id | The ID of the sent message | int |
| status | Status of the operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Send Telegram Photo

### What it is
Send a photo to a Telegram chat.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| chat_id | The chat ID to send the photo to | int | Yes |
| photo | Photo to send (URL, data URI, or workspace:// reference). URLs are preferred as Telegram will fetch them directly. | str (file) | Yes |
| caption | Caption for the photo (max 1024 characters) | str | No |
| parse_mode | Caption formatting mode | "none" \| "Markdown" \| "MarkdownV2" \| "HTML" | No |
| reply_to_message_id | Message ID to reply to | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_id | The ID of the sent message | int |
| status | Status of the operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Send Telegram Video

### What it is
Send a video to a Telegram chat.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| chat_id | The chat ID to send the video to | int | Yes |
| video | Video to send (MP4 format). Can be URL, data URI, or workspace:// reference. | str (file) | Yes |
| caption | Caption for the video | str | No |
| parse_mode | Caption formatting mode | "none" \| "Markdown" \| "MarkdownV2" \| "HTML" | No |
| duration | Duration in seconds | int | No |
| reply_to_message_id | Message ID to reply to | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_id | The ID of the sent message | int |
| status | Status of the operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Send Telegram Voice

### What it is
Send a voice message to a Telegram chat. Voice must be OGG format with OPUS codec.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| chat_id | The chat ID to send the voice message to | int | Yes |
| voice | Voice message to send (OGG format with OPUS codec). Can be URL, data URI, or workspace:// reference. | str (file) | Yes |
| caption | Caption for the voice message | str | No |
| duration | Duration in seconds | int | No |
| reply_to_message_id | Message ID to reply to | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_id | The ID of the sent message | int |
| status | Status of the operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
