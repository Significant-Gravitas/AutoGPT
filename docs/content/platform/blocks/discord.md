## Read Discord Messages

### What it is
A block that reads messages from a Discord channel using a bot token.

### What it does
This block connects to Discord using a bot token and retrieves messages from a specified channel. It can operate continuously or retrieve a single message.

### How it works
The block uses a Discord bot to log into a server and listen for new messages. When a message is received, it extracts the content, channel name, and username of the sender. If the message contains a text file attachment, the block also retrieves and includes the file's content.

### Inputs
| Input | Description |
|-------|-------------|
| Discord Bot Token | A secret token used to authenticate the bot with Discord |
| Continuous Read | A boolean flag indicating whether to continuously read messages or stop after one message |

### Outputs
| Output | Description |
|--------|-------------|
| Message Content | The text content of the received message, including any attached file content |
| Channel Name | The name of the Discord channel where the message was received |
| Username | The name of the user who sent the message |

### Possible use case
This block could be used to monitor a Discord channel for support requests. When a user posts a message, the block captures it, allowing another part of the system to process and respond to the request.

---

## Send Discord Message

### What it is
A block that sends messages to a Discord channel using a bot token.

### What it does
This block connects to Discord using a bot token and sends a specified message to a designated channel.

### How it works
The block uses a Discord bot to log into a server, locate the specified channel, and send the provided message. If the message is longer than Discord's character limit, it automatically splits the message into smaller chunks and sends them sequentially.

### Inputs
| Input | Description |
|-------|-------------|
| Discord Bot Token | A secret token used to authenticate the bot with Discord |
| Message Content | The text content of the message to be sent |
| Channel Name | The name of the Discord channel where the message should be sent |

### Outputs
| Output | Description |
|--------|-------------|
| Status | A string indicating the result of the operation (e.g., "Message sent" or "Channel not found") |

### Possible use case
This block could be used as part of an automated notification system. For example, it could send alerts to a Discord channel when certain events occur in another system, such as when a new user signs up or when a critical error is detected.