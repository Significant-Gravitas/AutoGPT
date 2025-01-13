
# Discord Blocks Documentation

## Read Discord Messages Block

### What it is
A block that connects to Discord and reads messages from channels where the bot is present.

### What it does
Monitors Discord channels and captures messages, including both text content and attached files (txt or py files). It extracts the message content, channel name, and username of the sender.

### How it works
The block connects to Discord using a bot token, listens for new messages, and when a message is received, it captures the message details. If the message includes text file attachments, it also downloads and includes their content in the output.

### Inputs
- Discord Bot Token: The authentication token for your Discord bot, required to connect to Discord servers

### Outputs
- Message Content: The text content of the received message, including any attached file contents
- Channel Name: The name of the channel where the message was received
- Username: The name of the user who sent the message

### Possible use case
Creating a logging system that archives all messages from specific Discord channels, or building a Discord-based support system that processes user requests.

## Send Discord Message Block

### What it is
A block that sends messages to specific Discord channels using a bot account.

### What it does
Connects to Discord and sends messages to specified channels. It can handle long messages by automatically splitting them into smaller chunks to comply with Discord's message length limitations.

### How it works
The block uses a Discord bot token to authenticate, locates the specified channel by name, and sends the message. If the message is longer than Discord's 2000-character limit, it automatically splits it into smaller parts and sends them sequentially.

### Inputs
- Discord Bot Token: The authentication token for your Discord bot
- Message Content: The text content you want to send
- Channel Name: The name of the channel where you want to send the message

### Outputs
- Status: The result of the operation (e.g., "Message sent" or "Channel not found")

### Possible use case
Creating automated notifications for a team's Discord server, such as sending project updates, alerts, or scheduled announcements. Could also be used to build a bridge between different communication platforms by forwarding messages to Discord.

