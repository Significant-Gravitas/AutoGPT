

## Read Discord Messages

### What it is
A block that monitors and captures messages from a Discord channel using a bot token.

### What it does
This block connects to Discord and reads incoming messages from specified channels, including the ability to process text file attachments.

### How it works
The block establishes a connection to Discord using a bot token, listens for new messages in the channels where the bot is present, and captures the message content, channel name, and username of the sender. If a message contains a text file attachment, it will also read and include the file's contents.

### Inputs
- Discord Credentials: A bot token that allows the block to connect to Discord servers

### Outputs
- Message Content: The actual text content of the received message, including any processed text file attachments
- Channel Name: The name of the Discord channel where the message was sent
- Username: The name of the user who sent the message

### Possible use case
A community manager wanting to monitor discussions in a Discord server and automatically collect messages for analysis or archiving. This could be particularly useful for gathering feedback, maintaining records of discussions, or tracking user engagement.

## Send Discord Message

### What it is
A block that sends messages to specific Discord channels using a bot token.

### What it does
This block enables automated message sending to Discord channels, with the ability to handle long messages by automatically splitting them into appropriate sizes.

### How it works
The block connects to Discord using a bot token, locates the specified channel by name, and sends the provided message. If the message is longer than Discord's character limit (2000 characters), it automatically splits the message into smaller chunks and sends them sequentially.

### Inputs
- Discord Credentials: A bot token that allows the block to connect to Discord servers
- Message Content: The text message that should be sent to Discord
- Channel Name: The name of the Discord channel where the message should be sent

### Outputs
- Status: A message indicating whether the operation was successful ("Message sent") or if there was an error

### Possible use case
An automated notification system that needs to send updates or alerts to a Discord community. For example, a project management system could use this block to automatically post status updates, announcements, or notifications about project milestones to a team's Discord channel.

