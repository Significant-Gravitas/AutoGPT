# Create Discord Thread

### What it is
Creates a new thread in a Discord channel.

### What it does
Creates a new thread in a Discord channel.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| channel_name | Channel ID or channel name to create the thread in | str | Yes |
| server_name | Server name (only needed if using channel name) | str | No |
| thread_name | The name of the thread to create | str | Yes |
| is_private | Whether to create a private thread (requires Boost Level 2+) or public thread | bool | No |
| auto_archive_duration | Duration before the thread is automatically archived | "60" | "1440" | "4320" | No |
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Discord Channel Info

### What it is
Resolves Discord channel names to IDs and vice versa.

### What it does
Resolves Discord channel names to IDs and vice versa.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Discord User Info

### What it is
Gets information about a Discord user by their ID.

### What it does
Gets information about a Discord user by their ID.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Read Discord Messages

### What it is
Reads messages from a Discord channel using a bot token.

### What it does
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

### What it does
Replies to a specific Discord message.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Send Discord DM

### What it is
Sends a direct message to a Discord user using their user ID.

### What it does
Sends a direct message to a Discord user using their user ID.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Send Discord Embed

### What it is
Sends a rich embed message to a Discord channel.

### What it does
Sends a rich embed message to a Discord channel.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| fields | List of field dictionaries with 'name', 'value', and optional 'inline' keys | List[Dict[str, True]] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| status | Operation status | str |
| message_id | ID of the sent embed message | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Send Discord File

### What it is
Sends a file attachment to a Discord channel.

### What it does
Sends a file attachment to a Discord channel.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Send Discord Message

### What it is
Sends a message to a Discord channel using a bot token.

### What it does
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
