
# Discord Blocks Documentation

## Read Discord Messages

### What it is
A block that monitors and retrieves messages from a Discord channel using a bot token.

### What it does
This block connects to Discord and listens for new messages in channels where the bot is present. It can capture text messages and handle file attachments (specifically .txt and .py files).

### How it works
The block connects to Discord using a bot token, monitors messages in real-time, and captures the message content, channel name, and username of the sender. When it receives a message with a text file attachment, it also downloads and includes the file's content in the output.

### Inputs
- Discord Credentials: A Discord bot token needed to authenticate and connect to Discord servers

### Outputs
- Message Content: The actual text content of the received message, including any attached file contents
- Channel Name: The name of the Discord channel where the message was received
- Username: The name of the user who sent the message

### Possible use case
Monitoring a support channel in Discord to automatically log customer inquiries or creating an archive of discussions in a community channel.

## Send Discord Message

### What it is
A block that sends messages to specific Discord channels using a bot token.

### What it does
This block enables automated message sending to designated Discord channels, handling long messages by automatically splitting them into appropriate sizes.

### How it works
The block connects to Discord using the provided bot token, locates the specified channel, and sends the message. If the message is longer than Discord's 2000-character limit, it automatically splits the message into smaller chunks and sends them sequentially.

### Inputs
- Discord Credentials: A Discord bot token needed to authenticate and connect to Discord servers
- Message Content: The text content you want to send to Discord
- Channel Name: The name of the Discord channel where you want to send the message

### Outputs
- Status: The result of the operation (e.g., "Message sent" or error messages if something goes wrong)

### Possible use case
Automatically sending notifications to a Discord channel when certain events occur, such as system alerts, user notifications, or automated reports.

---

**Note**: Both blocks require valid Discord bot credentials with appropriate permissions to function correctly. The bot needs to be added to the Discord server where it will operate.
