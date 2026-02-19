# Telegram Blocks
<!-- MANUAL: file_description -->
These blocks let your agent interact with Telegram through the Bot API. They cover sending text, photos, video, audio, voice messages, and documents, as well as replying to, editing, and deleting messages. Media blocks accept URLs (passed directly to Telegram for server-side fetch), data URIs, and `workspace://` references (resolved locally and uploaded via multipart form-data). All blocks require a Telegram Bot API token obtained from [@BotFather](https://t.me/BotFather).
<!-- END MANUAL -->

## Delete Telegram Message

### What it is
Delete a message from a Telegram chat. Bots can delete their own messages and incoming messages in private chats at any time.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Telegram Bot API `deleteMessage` method with the provided `chat_id` and `message_id`. On success, outputs a status confirmation. Note that bots can only delete their own messages in any chat, or incoming messages in private chats. In groups, deleting other users' messages requires admin privileges.
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
Automatically clean up expired notifications or temporary status messages sent by your bot. For example, after a user confirms an action, delete the original prompt message to keep the chat tidy.
<!-- END MANUAL -->

---

## Edit Telegram Message

### What it is
Edit the text of an existing message sent by the bot.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Telegram Bot API `editMessageText` method with the target `chat_id`, `message_id`, and the new `text`. An optional `parse_mode` can be set to format the replacement text as Markdown or HTML. Only messages sent by the bot itself can be edited.
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
Update a "Processing..." status message with the final result once a long-running task completes, so the user sees progress in-place rather than receiving a separate follow-up message.
<!-- END MANUAL -->

---

## Get Telegram File

### What it is
Download a file from Telegram using its file_id. Use this to process photos, voice messages, or documents received.

### How it works
<!-- MANUAL: how_it_works -->
First calls the `getFile` API method to resolve the `file_id` into a server-side file path, then downloads the raw bytes from Telegram's file server. The downloaded content is converted to a data URI and stored via the workspace file system, outputting a `workspace://` reference (or data URI) that other blocks can consume.
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
Download a photo sent by a user and pass it to an image recognition or OCR block for processing, then reply with the extracted information.
<!-- END MANUAL -->

---

## Reply To Telegram Message

### What it is
Reply to a specific message in a Telegram chat.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Telegram Bot API `sendMessage` method with a `reply_to_message_id` parameter, which creates a new message visually linked to the original. The reply appears with a quoted preview of the original message in the chat. An optional `parse_mode` enables Markdown or HTML formatting.
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
In a group chat, reply directly to a user's question with an AI-generated answer so that the response is clearly threaded to the original question, keeping the conversation organized.
<!-- END MANUAL -->

---

## Send Telegram Audio

### What it is
Send an audio file to a Telegram chat. The file is displayed in the music player. For voice messages, use the Send Voice block instead.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Telegram Bot API `sendAudio` method. If the input is an HTTP(S) URL, it is passed directly to Telegram which fetches the file server-side. For data URIs or `workspace://` references, the file is resolved locally and uploaded via multipart form-data. Optional metadata like `title`, `performer`, and `duration` is included when provided.
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
Send a text-to-speech audio file generated by an AI model back to the user as a playable track, complete with a title and caption describing what was generated.
<!-- END MANUAL -->

---

## Send Telegram Document

### What it is
Send a document (any file type) to a Telegram chat.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Telegram Bot API `sendDocument` method. URLs are passed directly for server-side fetch; data URIs and `workspace://` references are resolved locally and uploaded via multipart form-data. A custom `filename` can be specified to control the display name shown to the recipient. The caption supports optional Markdown or HTML formatting via `parse_mode`.
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
Generate a PDF report from collected data and send it to the user as a downloadable file with a descriptive filename like `weekly-report.pdf`.
<!-- END MANUAL -->

---

## Send Telegram Message

### What it is
Send a text message to a Telegram chat.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Telegram Bot API `sendMessage` method with the provided `chat_id` and `text`. Optional parameters include `parse_mode` (to render Markdown or HTML formatting), `reply_to_message_id` (to thread the message as a reply), and `disable_notification` (to send silently without triggering a sound on the recipient's device).
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
Build a conversational bot that receives a user's question via the Message Trigger, processes it through an AI block, and sends the answer back using this block. Use `parse_mode` to format responses with bold headings or code blocks.
<!-- END MANUAL -->

---

## Send Telegram Photo

### What it is
Send a photo to a Telegram chat.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Telegram Bot API `sendPhoto` method. If the input is an HTTP(S) URL, it is passed directly to Telegram which fetches the image server-side (preferred for speed and efficiency). For data URIs or `workspace://` references, the file is resolved to a local path, read as bytes, and uploaded via multipart form-data with the appropriate MIME type.
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
Send an AI-generated image (e.g., from DALL-E or Stable Diffusion) back to the user who requested it, with a caption describing the prompt used.
<!-- END MANUAL -->

---

## Send Telegram Video

### What it is
Send a video to a Telegram chat.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Telegram Bot API `sendVideo` method. URLs are passed directly for server-side fetch; data URIs and `workspace://` references are resolved locally and uploaded via multipart form-data with MIME type detection. Optional `duration` metadata and `parse_mode` for the caption can be provided.
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
Send a tutorial or demo video clip in response to a user's help request, or deliver a dynamically generated video summary of data trends.
<!-- END MANUAL -->

---

## Send Telegram Voice

### What it is
Send a voice message to a Telegram chat. Voice must be OGG format with OPUS codec.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Telegram Bot API `sendVoice` method. URLs are passed directly for server-side fetch; data URIs and `workspace://` references are resolved locally and uploaded via multipart form-data. The file must be in OGG format encoded with the OPUS codec for Telegram to display it as a voice message (with a waveform). Other formats will not render correctly as voice messages.
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
Convert an AI-generated text response to speech using a TTS block and send it as a voice message, creating a voice-based conversational assistant.
<!-- END MANUAL -->

---
