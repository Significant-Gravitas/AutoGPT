
# Discord Integration Blocks

## Read Discord Messages

### What it is
A block that monitors and captures messages from a Discord channel using a bot token.

### What it does
This block connects to Discord and listens for new messages in channels where the bot is present. It can capture regular text messages and also handle file attachments (text and Python files).

### How it works
The block establishes a connection to Discord using a bot token, waits for new messages, and when a message is received, it processes both the message content and any attached files. If a text file is attached, it will read and include its contents in the output.

### Inputs
- Discord Credentials: A bot token that allows the block to connect to Discord servers and read messages

### Outputs
- Message Content: The actual text content of the received message, including any processed file attachments
- Channel Name: The name of the Discord channel where the message was received
- Username: The name of the user who sent the message

### Possible use case
A company wants to monitor their Discord support channel and automatically process customer inquiries. This block could capture the messages and feed them into other blocks for automated response generation or ticket creation.

## Send Discord Message

### What it is
A block that sends messages to specific Discord channels using a bot token.

### What it does
This block enables automated message sending to Discord channels. It can handle long messages by automatically splitting them into appropriate sizes that Discord can process.

### How it works
The block connects to Discord using a bot token, locates the specified channel, and sends the provided message. If the message is too long, it automatically splits it into smaller chunks to comply with Discord's message size limits.

### Inputs
- Discord Credentials: A bot token that allows the block to connect to Discord servers and send messages
- Message Content: The text content you want to send to Discord
- Channel Name: The name of the Discord channel where you want to send the message

### Outputs
- Status: A message indicating whether the operation was successful ("Message sent") or if there was an error

### Possible use case
An automated notification system that needs to send updates to a Discord channel. For example, a development team could use this block to automatically post deployment status updates or system alerts to their Discord server.

