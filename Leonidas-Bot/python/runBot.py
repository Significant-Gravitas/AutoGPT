# Built-in imports
import os
import json
from dotenv import load_dotenv

# Local imports
import utils
import utils_grpc
from messageTypes import MessageTypes

# Third-party imports
import discord

load_dotenv()

# Necessary global variables
ENABLE_VOICE_RESPONSE = os.getenv("ENABLE_VOICE_RESPONSE").lower() == "false"
CONVAI_CHARACTER_ID = os.getenv("CONVAI_CHARACTER_ID")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

ALLOWED_CHANNELS = os.getenv("ALLOWED_CHANNELS").split(",") if os.getenv("ALLOWED_CHANNELS") else None
SESSION_IDS = {}    # Maintaining separate sessions of conversation for separate channels and private messages

# Send messages
async def sendMessage(message, user_message, is_private):
    global SESSION_IDS
    global ENABLE_VOICE_RESPONSE
    global CONVAI_CHARACTER_ID

    try:
        # Check if the channel is listed in SESSION_IDS to maintain context of conversation
        sessionID = SESSION_IDS.setdefault(message.channel.id, "-1")

        response = utils_grpc.getResponseGRPC(
            userQuery=user_message,
            sessionID= sessionID,
            voiceResponse=ENABLE_VOICE_RESPONSE
        )

        # Updating SESSION_ID with new value for new chat sessions after it has been set to -1
        if sessionID == "-1":
            SESSION_IDS[message.channel.id] = response["sessionID"]

        # Replying with the required text receive from the api call
        if is_private:
            await message.author.send(response["text"])
        else:
            await message.channel.send(response["text"])

    except Exception as e:
        print(e)

def runDiscordBot():
    global ALLOWED_CHANNELS
    global SESSION_IDS

    client = discord.Client(
        intents=discord.Intents.all()
    )

    @client.event
    async def on_ready():
        print(f'{client.user} is now running!')

    @client.event
    async def on_message(message):
        # Make sure bot doesn't get stuck in an infinite loop
        if message.author == client.user:
            return

        # Get data about the user
        username = str(message.author)
        user_message = str(message.content)
        channel = str(message.channel)

        # Restricting the bot to speak only in a specific channel and DM channels
        if type(message.channel) != discord.DMChannel and ALLOWED_CHANNELS and message.channel.name not in ALLOWED_CHANNELS:
            return

        # Debug printing
        print(f"{username} said: '{user_message}' ({channel})")
        # Parsing the message 
        user_message_suffix, message_type = utils.parse_message(user_message, client.user.id)

        #If the message is a direct message, you do not need any prefix to talk with the chatbot
        if type(message.channel) == discord.DMChannel:

            if message_type == MessageTypes.CHAT_RESET:
                SESSION_IDS[message.channel.id] = "-1"
                await message.channel.send("I just had a great sleep. Feeling refreshed and better.")

            elif message_type == MessageTypes.NO_RESPONSE:
                return

            else:
                await sendMessage(message, user_message_suffix, is_private=True)

        # For all text-channel messages
        else:
            # if message_type == MessageTypes.CHANNEL_MENTION:
            #     await sendMessage(message, user_message, is_private=False)

            if message_type == MessageTypes.GO_PRIVATE:
                # Advisable to not use this code right now. Needs extensive testing.
                await sendMessage(message, user_message_suffix, is_private=True)

            elif message_type == MessageTypes.CHAT_RESET:
                SESSION_IDS[message.channel.id] = "-1"
                await message.channel.send("I just had a great sleep. Feeling refreshed and better.")

            elif message_type == MessageTypes.RESPOND:
                await sendMessage(message, user_message_suffix, is_private=False)

            else:
                # Do not reply to messages you are not mentioned
                return

    # Remember to run your bot with your personal TOKEN
    client.run(DISCORD_BOT_TOKEN)