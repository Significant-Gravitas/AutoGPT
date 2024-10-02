import asyncio

import aiohttp
import discord
from pydantic import Field

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SecretField


class ReadDiscordMessagesBlock(Block):
    class Input(BlockSchema):
        discord_bot_token: BlockSecret = SecretField(
            key="discord_bot_token", description="Discord bot token"
        )
        continuous_read: bool = Field(
            description="Whether to continuously read messages", default=True
        )

    class Output(BlockSchema):
        message_content: str = Field(description="The content of the message received")
        channel_name: str = Field(
            description="The name of the channel the message was received from"
        )
        username: str = Field(
            description="The username of the user who sent the message"
        )

    def __init__(self):
        super().__init__(
            id="df06086a-d5ac-4abb-9996-2ad0acb2eff7",
            input_schema=ReadDiscordMessagesBlock.Input,  # Assign input schema
            output_schema=ReadDiscordMessagesBlock.Output,  # Assign output schema
            description="Reads messages from a Discord channel using a bot token.",
            categories={BlockCategory.SOCIAL},
            test_input={"discord_bot_token": "test_token", "continuous_read": False},
            test_output=[
                (
                    "message_content",
                    "Hello!\n\nFile from user: example.txt\nContent: This is the content of the file.",
                ),
                ("channel_name", "general"),
                ("username", "test_user"),
            ],
            test_mock={
                "run_bot": lambda token: asyncio.Future()  # Create a Future object for mocking
            },
        )

    async def run_bot(self, token: str):
        intents = discord.Intents.default()
        intents.message_content = True

        client = discord.Client(intents=intents)

        self.output_data = None
        self.channel_name = None
        self.username = None

        @client.event
        async def on_ready():
            print(f"Logged in as {client.user}")

        @client.event
        async def on_message(message):
            if message.author == client.user:
                return

            self.output_data = message.content
            self.channel_name = message.channel.name
            self.username = message.author.name

            if message.attachments:
                attachment = message.attachments[0]  # Process the first attachment
                if attachment.filename.endswith((".txt", ".py")):
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as response:
                            file_content = await response.text()
                            self.output_data += f"\n\nFile from user: {attachment.filename}\nContent: {file_content}"

            await client.close()

        await client.start(token)

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        while True:
            for output_name, output_value in self.__run(input_data):
                yield output_name, output_value
            if not input_data.continuous_read:
                break

    def __run(self, input_data: Input) -> BlockOutput:
        try:
            loop = asyncio.get_event_loop()
            future = self.run_bot(input_data.discord_bot_token.get_secret_value())

            # If it's a Future (mock), set the result
            if isinstance(future, asyncio.Future):
                future.set_result(
                    {
                        "output_data": "Hello!\n\nFile from user: example.txt\nContent: This is the content of the file.",
                        "channel_name": "general",
                        "username": "test_user",
                    }
                )

            result = loop.run_until_complete(future)

            # For testing purposes, use the mocked result
            if isinstance(result, dict):
                self.output_data = result.get("output_data")
                self.channel_name = result.get("channel_name")
                self.username = result.get("username")

            if (
                self.output_data is None
                or self.channel_name is None
                or self.username is None
            ):
                raise ValueError("No message, channel name, or username received.")

            yield "message_content", self.output_data
            yield "channel_name", self.channel_name
            yield "username", self.username

        except discord.errors.LoginFailure as login_err:
            raise ValueError(f"Login error occurred: {login_err}")
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")


class SendDiscordMessageBlock(Block):
    class Input(BlockSchema):
        discord_bot_token: BlockSecret = SecretField(
            key="discord_bot_token", description="Discord bot token"
        )
        message_content: str = Field(description="The content of the message received")
        channel_name: str = Field(
            description="The name of the channel the message was received from"
        )

    class Output(BlockSchema):
        status: str = Field(
            description="The status of the operation (e.g., 'Message sent', 'Error')"
        )

    def __init__(self):
        super().__init__(
            id="d0822ab5-9f8a-44a3-8971-531dd0178b6b",
            input_schema=SendDiscordMessageBlock.Input,  # Assign input schema
            output_schema=SendDiscordMessageBlock.Output,  # Assign output schema
            description="Sends a message to a Discord channel using a bot token.",
            categories={BlockCategory.SOCIAL},
            test_input={
                "discord_bot_token": "YOUR_DISCORD_BOT_TOKEN",
                "channel_name": "general",
                "message_content": "Hello, Discord!",
            },
            test_output=[("status", "Message sent")],
            test_mock={
                "send_message": lambda token, channel_name, message_content: asyncio.Future()
            },
        )

    async def send_message(self, token: str, channel_name: str, message_content: str):
        intents = discord.Intents.default()
        intents.guilds = True  # Required for fetching guild/channel information
        client = discord.Client(intents=intents)

        @client.event
        async def on_ready():
            print(f"Logged in as {client.user}")
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
        return [message[i : i + limit] for i in range(0, len(message), limit)]

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            loop = asyncio.get_event_loop()
            future = self.send_message(
                input_data.discord_bot_token.get_secret_value(),
                input_data.channel_name,
                input_data.message_content,
            )

            # If it's a Future (mock), set the result
            if isinstance(future, asyncio.Future):
                future.set_result("Message sent")

            result = loop.run_until_complete(future)

            # For testing purposes, use the mocked result
            if isinstance(result, str):
                self.output_data = result

            if self.output_data is None:
                raise ValueError("No status message received.")

            yield "status", self.output_data

        except discord.errors.LoginFailure as login_err:
            raise ValueError(f"Login error occurred: {login_err}")
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")
