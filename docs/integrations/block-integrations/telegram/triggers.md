# Telegram Triggers
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Telegram Message Reaction Trigger

### What it is
Triggers when a reaction to a message is changed. Works in private chats automatically. In groups, the bot must be an administrator.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Telegram Message Trigger

### What it is
Triggers when a message is received by your Telegram bot. Supports text, photos, voice messages, and audio files.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
