import discord
import asyncio
from autogpt_server.data.block import Block, BlockSchema, BlockOutput
import aiohttp

class DiscordBot(Block):
    class Input(BlockSchema):
        token: str  # The token for the Discord bot

    class Output(BlockSchema):
        message_content: str  # The content of the message received
        channel_name: str  # The name of the channel the message was received from

    def __init__(self):
        super().__init__(
            id="d3f4g5h6-1i2j-3k4l-5m6n-7o8p9q0r1s2t",  # Unique ID for the node
            input_schema=DiscordBot.Input,  # Assign input schema
            output_schema=DiscordBot.Output,  # Assign output schema
            test_input={"token": "YOUR_DISCORD_BOT_TOKEN"},
            test_output={"message_content": "Hello!\n\nFile from user: example.txt\nContent: This is the content of the file.", "channel_name": "general"},
        )

    async def run_bot(self, token: str):
        intents = discord.Intents.default()
        intents.message_content = True

        client = discord.Client(intents=intents)

        self.output_data = None
        self.channel_name = None

        @client.event
        async def on_ready():
            print(f'Logged in as {client.user}')

        @client.event
        async def on_message(message):
            if message.author == client.user:
                return

            self.output_data = message.content
            self.channel_name = message.channel.name

            if message.attachments:
                attachment = message.attachments[0]  # Process the first attachment
                if attachment.filename.endswith(('.txt', '.py')):
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as response:
                            file_content = await response.text()
                            self.output_data += f"\n\nFile from user: {attachment.filename}\nContent: {file_content}"

            await client.close()

        await client.start(token)

    def run(self, input_data: 'DiscordBot.Input') -> BlockOutput:
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.run_bot(input_data.token))

            if self.output_data is None or self.channel_name is None:
                raise ValueError("No message or channel name received.")

            yield "message_content", self.output_data
            yield "channel_name", self.channel_name

        except discord.errors.LoginFailure as login_err:
            raise ValueError(f"Login error occurred: {login_err}")
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")


class DiscordMessageSender(Block):
    class Input(BlockSchema):
        token: str  # The token for the Discord bot
        channel_name: str  # The name of the channel to send the message to
        message_content: str  # The content of the message to send

    class Output(BlockSchema):
        status: str  # The status of the operation (e.g., 'Message sent', 'Error')

    def __init__(self):
        super().__init__(
            id="h1i2j3k4-5l6m-7n8o-9p0q-r1s2t3u4v5w6",  # Unique ID for the node
            input_schema=DiscordMessageSender.Input,  # Assign input schema
            output_schema=DiscordMessageSender.Output,  # Assign output schema
            test_input={
                "token": "YOUR_DISCORD_BOT_TOKEN",
                "channel_name": "general",
                "message_content": "Hello, Discord!"
            },
            test_output={"status": "Message sent"},
        )

    async def send_message(self, token: str, channel_name: str, message_content: str):
        intents = discord.Intents.default()
        intents.guilds = True  # Required for fetching guild/channel information
        client = discord.Client(intents=intents)

        @client.event
        async def on_ready():
            print(f'Logged in as {client.user}')
            for guild in client.guilds:
                for channel in guild.text_channels:
                    if channel.name == channel_name:
                        # Split message into chunks if it exceeds 2000 characters
                        for chunk in self.chunk_message(message_content):
                            await channel.send(chunk)
                        self.output_data = "Message sent"
                        await client.close()
                        return

            self.output_data = "Channel not found"
            await client.close()

        await client.start(token)

    def chunk_message(self, message: str, limit: int = 2000) -> list:
        """Splits a message into chunks not exceeding the Discord limit."""
        return [message[i:i + limit] for i in range(0, len(message), limit)]

    def run(self, input_data: 'DiscordMessageSender.Input') -> BlockOutput:
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(
                self.send_message(input_data.token, input_data.channel_name, input_data.message_content)
            )

            if self.output_data is None:
                raise ValueError("No status message received.")

            yield "status", self.output_data

        except discord.errors.LoginFailure as login_err:
            raise ValueError(f"Login error occurred: {login_err}")
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")
