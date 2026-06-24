# Telegram Triggers
<!-- MANUAL: file_description -->
These trigger blocks let your agent receive incoming messages and reactions from Telegram in real time via webhooks. When a user sends a message or reacts to one, the trigger fires and outputs structured data (chat ID, user info, message content, file IDs) that downstream blocks can process.
<!-- END MANUAL -->

## Telegram Message Reaction Trigger

### What it is
Triggers when a reaction to a message is changed. Works in private chats automatically. In groups, the bot must be an administrator.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Telegram Bot API webhook system, subscribing to `message_reaction` updates. When a user adds, changes, or removes a reaction on a message in a chat with your bot, Telegram sends an update to the registered webhook URL. The block extracts the chat ID, message ID, reacting user's info, and both the old and new reaction lists. In private chats this works automatically; in group chats the bot must be an administrator to receive reaction updates.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| payload | The complete webhook payload from Telegram | Dict[str, Any] |
| chat_id | The chat ID where the reaction occurred | int |
| message_id | The message ID that was reacted to | int |
| user_id | The user ID who changed the reaction | int |
| username | Username of the user (may be empty) | str |
| new_reactions | List of new reactions on the message | List[Any] |
| old_reactions | List of previous reactions on the message | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Sentiment tracking** — Monitor reactions on bot-posted announcements to gauge audience sentiment in real time.

**Approval workflows** — Use a thumbs-up reaction as a lightweight approval signal to trigger downstream actions like deployments or task assignments.

**Engagement analytics** — Aggregate reaction data across messages to identify which content resonates most with your audience.
<!-- END MANUAL -->

---

## Telegram Message Trigger

### What it is
Triggers when a message is received or edited in your Telegram bot. Supports text, photos, voice messages, audio files, documents, and videos.

### How it works
<!-- MANUAL: how_it_works -->
This block registers a webhook with the Telegram Bot API that subscribes to `message` and `edited_message` updates. Incoming messages are routed by content type — text, photo, voice, audio, document, or video — based on the event filter you configure. When a matching message arrives, the block extracts common fields (chat ID, sender info, message ID) along with type-specific data such as the text content, file IDs for media, or captions. File IDs can be passed to the Get Telegram File block to download the actual media. If the "edited_message" event is enabled, the block also fires when a user edits a previously sent message, with the `is_edited` output set to `true`.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| events | Types of messages to receive | Message Types | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| payload | The complete webhook payload from Telegram | Dict[str, Any] |
| chat_id | The chat ID where the message was received. Use this to send replies. | int |
| message_id | The unique message ID | int |
| user_id | The user ID who sent the message | int |
| username | Username of the sender (may be empty) | str |
| first_name | First name of the sender | str |
| event | The message type (text, photo, voice, audio, etc.) | str |
| text | Text content of the message (for text messages) | str |
| photo_file_id | File ID of the photo (for photo messages). Use GetTelegramFileBlock to download. | str |
| voice_file_id | File ID of the voice message (for voice messages). Use GetTelegramFileBlock to download. | str |
| audio_file_id | File ID of the audio file (for audio messages). Use GetTelegramFileBlock to download. | str |
| file_id | File ID for document/video messages. Use GetTelegramFileBlock to download. | str |
| file_name | Original filename (for document/audio messages) | str |
| caption | Caption for media messages | str |
| is_edited | Whether this is an edit of a previously sent message | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Conversational AI bot** — Receive text messages from users and feed them into an AI agent that generates and sends replies.

**Photo processing pipeline** — Trigger on incoming photos, download them with Get Telegram File, run image analysis or OCR, and reply with the results.

**Voice message transcription** — Capture voice messages, download the audio file, pass it to a speech-to-text service, and send the transcript back to the user.
<!-- END MANUAL -->

---
