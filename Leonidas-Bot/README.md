# Convai_Discord_Integration
This repository contains the required code to facilitate integration of Convai chat apis with Discord bot for smooth integration.

V0.1.0
Contains a simple text-in-text-out character chatbot implementation with Convai.
Languages: Python

Steps to follow:
- Navigate to the `python` folder.
- Open the `.env` file and add the Discord Bot Token and your Convai API Key
- Replace the CONVAI-CHARACTER-ID value with your own character id. By default it is populated with the ID of a sample character.
- We are keeping audio response disabled in this version
- Next, open terminal and run `python main.py`

Note: The steps to create a discord bot and add them to a server is not being defined here. Please refer docs for more info.

V0.1.1

Improvements:
- Every conversation directed to the chatbot in the channel needs to start by mentioning the bot in the message. So the format stands: `@ConvaiBot <message>`
- Now you can reset the chatbot in a particular channel by typing `/reset @ConvaiBot`. This sets the session-id of the conversation in the channel back to "-1", indicating the start of a new conversation.
- You can also take a conversation with the Chatbot to DM, by typing `/private @ConvaiBot <message>` and this will start a new session of conversation with the given message in Private Message with the bot.
- All conversation context maintenance is being done with channel-id of the messages being received
- Removed default-character from env variables. Please start with a new one.
Note: DM Channel / Private Messages to the bot is in beta mode.

V0.1.2

Improvements:
- Major code refactoring to include upcoming features.
- Added simple unit-testing code for python bot. 

Note: Run `python -m pytest test` inside the `python` folder.

#### Docker Deployment

Commands to run the server inside a docker container :
- `cd python`
- `sudo docker build -t convai_discord_server .`
- `sudo docker run convai_discord_server`