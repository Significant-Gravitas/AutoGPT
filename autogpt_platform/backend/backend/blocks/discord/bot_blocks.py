import base64
import io
import mimetypes
from pathlib import Path
from typing import Any

import discord
from pydantic import SecretStr

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import APIKeyCredentials, SchemaField
from backend.util.file import store_media_file
from backend.util.request import Requests
from backend.util.type import MediaFileType

from ._auth import (
    TEST_BOT_CREDENTIALS,
    TEST_BOT_CREDENTIALS_INPUT,
    DiscordBotCredentialsField,
    DiscordBotCredentialsInput,
)

# Keep backward compatibility alias
DiscordCredentials = DiscordBotCredentialsInput
DiscordCredentialsField = DiscordBotCredentialsField
TEST_CREDENTIALS = TEST_BOT_CREDENTIALS
TEST_CREDENTIALS_INPUT = TEST_BOT_CREDENTIALS_INPUT


class ReadDiscordMessagesBlock(Block):
    class Input(BlockSchema):
        credentials: DiscordCredentials = DiscordCredentialsField()

    class Output(BlockSchema):
        message_content: str = SchemaField(
            description="The content of the message received"
        )
        message_id: str = SchemaField(description="The ID of the message")
        channel_id: str = SchemaField(description="The ID of the channel")
        channel_name: str = SchemaField(
            description="The name of the channel the message was received from"
        )
        user_id: str = SchemaField(
            description="The ID of the user who sent the message"
        )
        username: str = SchemaField(
            description="The username of the user who sent the message"
        )

    def __init__(self):
        super().__init__(
            id="df06086a-d5ac-4abb-9996-2ad0acb2eff7",
            input_schema=ReadDiscordMessagesBlock.Input,  # Assign input schema
            output_schema=ReadDiscordMessagesBlock.Output,  # Assign output schema
            description="Reads messages from a Discord channel using a bot token.",
            categories={BlockCategory.SOCIAL},
            test_input={
                "continuous_read": False,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "message_content",
                    "Hello!\n\nFile from user: example.txt\nContent: This is the content of the file.",
                ),
                ("message_id", "123456789012345678"),
                ("channel_id", "987654321098765432"),
                ("channel_name", "general"),
                ("user_id", "111222333444555666"),
                ("username", "test_user"),
            ],
            test_mock={
                "run_bot": lambda token: {
                    "output_data": "Hello!\n\nFile from user: example.txt\nContent: This is the content of the file.",
                    "message_id": "123456789012345678",
                    "channel_id": "987654321098765432",
                    "channel_name": "general",
                    "user_id": "111222333444555666",
                    "username": "test_user",
                }
            },
        )

    async def run_bot(self, token: SecretStr):
        intents = discord.Intents.default()
        intents.message_content = True

        client = discord.Client(intents=intents)

        self.output_data = None
        self.message_id = None
        self.channel_id = None
        self.channel_name = None
        self.user_id = None
        self.username = None

        @client.event
        async def on_ready():
            print(f"Logged in as {client.user}")

        @client.event
        async def on_message(message):
            if message.author == client.user:
                return

            self.output_data = message.content
            self.message_id = str(message.id)
            self.channel_id = str(message.channel.id)
            self.channel_name = message.channel.name
            self.user_id = str(message.author.id)
            self.username = message.author.name

            if message.attachments:
                attachment = message.attachments[0]  # Process the first attachment
                if attachment.filename.endswith((".txt", ".py")):
                    response = await Requests().get(attachment.url)
                    file_content = response.text()
                    self.output_data += f"\n\nFile from user: {attachment.filename}\nContent: {file_content}"

            await client.close()

        await client.start(token.get_secret_value())

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        async for output_name, output_value in self.__run(input_data, credentials):
            yield output_name, output_value

    async def __run(
        self, input_data: Input, credentials: APIKeyCredentials
    ) -> BlockOutput:
        try:
            result = await self.run_bot(credentials.api_key)

            # For testing purposes, use the mocked result
            if isinstance(result, dict):
                self.output_data = result.get("output_data")
                self.message_id = result.get("message_id")
                self.channel_id = result.get("channel_id")
                self.channel_name = result.get("channel_name")
                self.user_id = result.get("user_id")
                self.username = result.get("username")

            if (
                self.output_data is None
                or self.channel_name is None
                or self.username is None
            ):
                raise ValueError("No message, channel name, or username received.")

            yield "message_content", self.output_data
            yield "message_id", self.message_id
            yield "channel_id", self.channel_id
            yield "channel_name", self.channel_name
            yield "user_id", self.user_id
            yield "username", self.username

        except discord.errors.LoginFailure as login_err:
            raise ValueError(f"Login error occurred: {login_err}")
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")


class SendDiscordMessageBlock(Block):
    class Input(BlockSchema):
        credentials: DiscordCredentials = DiscordCredentialsField()
        message_content: str = SchemaField(
            description="The content of the message to send"
        )
        channel_name: str = SchemaField(
            description="Channel ID or channel name to send the message to"
        )
        server_name: str = SchemaField(
            description="Server name (only needed if using channel name)",
            advanced=True,
            default="",
        )

    class Output(BlockSchema):
        status: str = SchemaField(
            description="The status of the operation (e.g., 'Message sent', 'Error')"
        )
        message_id: str = SchemaField(description="The ID of the sent message")
        channel_id: str = SchemaField(
            description="The ID of the channel where the message was sent"
        )

    def __init__(self):
        super().__init__(
            id="d0822ab5-9f8a-44a3-8971-531dd0178b6b",
            input_schema=SendDiscordMessageBlock.Input,  # Assign input schema
            output_schema=SendDiscordMessageBlock.Output,  # Assign output schema
            description="Sends a message to a Discord channel using a bot token.",
            categories={BlockCategory.SOCIAL},
            test_input={
                "channel_name": "general",
                "message_content": "Hello, Discord!",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("status", "Message sent"),
                ("message_id", "123456789012345678"),
                ("channel_id", "987654321098765432"),
            ],
            test_mock={
                "send_message": lambda token, channel_name, server_name, message_content: {
                    "status": "Message sent",
                    "message_id": "123456789012345678",
                    "channel_id": "987654321098765432",
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    async def send_message(
        self,
        token: str,
        channel_name: str,
        server_name: str | None,
        message_content: str,
    ) -> dict:
        intents = discord.Intents.default()
        intents.guilds = True  # Required for fetching guild/channel information
        client = discord.Client(intents=intents)

        result = {}

        @client.event
        async def on_ready():
            print(f"Logged in as {client.user}")
            channel = None

            # Try to parse as channel ID first
            try:
                channel_id = int(channel_name)
                channel = client.get_channel(channel_id)
            except ValueError:
                # Not a valid ID, will try name lookup
                pass

            # If not found by ID (or not an ID), try name lookup
            if not channel:
                for guild in client.guilds:
                    if server_name and guild.name != server_name:
                        continue
                    for ch in guild.text_channels:
                        if ch.name == channel_name:
                            channel = ch
                            break
                    if channel:
                        break

            if not channel:
                result["status"] = f"Channel not found: {channel_name}"
                await client.close()
                return

            # Type check - ensure it's a text channel that can send messages
            if not hasattr(channel, "send"):
                result["status"] = (
                    f"Channel {channel_name} cannot receive messages (not a text channel)"
                )
                await client.close()
                return

            # Split message into chunks if it exceeds 2000 characters
            chunks = self.chunk_message(message_content)
            last_message = None
            for chunk in chunks:
                last_message = await channel.send(chunk)  # type: ignore
            result["status"] = "Message sent"
            result["message_id"] = str(last_message.id) if last_message else ""
            result["channel_id"] = str(channel.id)
            await client.close()

        await client.start(token)
        return result

    def chunk_message(self, message: str, limit: int = 2000) -> list:
        """Splits a message into chunks not exceeding the Discord limit."""
        return [message[i : i + limit] for i in range(0, len(message), limit)]

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = await self.send_message(
                token=credentials.api_key.get_secret_value(),
                channel_name=input_data.channel_name,
                server_name=input_data.server_name,
                message_content=input_data.message_content,
            )

            # For testing purposes, use the mocked result
            if isinstance(result, str):
                result = {"status": result}

            yield "status", result.get("status", "Unknown error")
            if "message_id" in result:
                yield "message_id", result["message_id"]
            if "channel_id" in result:
                yield "channel_id", result["channel_id"]

        except discord.errors.LoginFailure as login_err:
            raise ValueError(f"Login error occurred: {login_err}")
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")


class SendDiscordDMBlock(Block):
    class Input(BlockSchema):
        credentials: DiscordCredentials = DiscordCredentialsField()
        user_id: str = SchemaField(
            description="The Discord user ID to send the DM to (e.g., '123456789012345678')"
        )
        message_content: str = SchemaField(
            description="The content of the direct message to send"
        )

    class Output(BlockSchema):
        status: str = SchemaField(description="The status of the operation")
        message_id: str = SchemaField(description="The ID of the sent message")

    def __init__(self):
        super().__init__(
            id="40d71a5a-e268-4060-9ee0-38ae6f225682",
            input_schema=SendDiscordDMBlock.Input,
            output_schema=SendDiscordDMBlock.Output,
            description="Sends a direct message to a Discord user using their user ID.",
            categories={BlockCategory.SOCIAL},
            test_input={
                "user_id": "123456789012345678",
                "message_content": "Hello! This is a test DM.",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("status", "DM sent successfully"),
                ("message_id", "987654321098765432"),
            ],
            test_mock={
                "send_dm": lambda token, user_id, message_content: {
                    "status": "DM sent successfully",
                    "message_id": "987654321098765432",
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    async def send_dm(self, token: str, user_id: str, message_content: str) -> dict:
        intents = discord.Intents.default()
        intents.dm_messages = True
        client = discord.Client(intents=intents)

        result = {}

        @client.event
        async def on_ready():
            try:
                user = await client.fetch_user(int(user_id))
                message = await user.send(message_content)
                result["status"] = "DM sent successfully"
                result["message_id"] = str(message.id)
            except discord.errors.Forbidden:
                result["status"] = (
                    "Cannot send DM - user has DMs disabled or bot is blocked"
                )
            except discord.errors.NotFound:
                result["status"] = f"User with ID {user_id} not found"
            except ValueError:
                result["status"] = f"Invalid user ID format: {user_id}"
            except Exception as e:
                result["status"] = f"Error sending DM: {str(e)}"
            finally:
                await client.close()

        await client.start(token)
        return result

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = await self.send_dm(
                token=credentials.api_key.get_secret_value(),
                user_id=input_data.user_id,
                message_content=input_data.message_content,
            )

            yield "status", result.get("status", "Unknown error")
            if "message_id" in result:
                yield "message_id", result["message_id"]

        except discord.errors.LoginFailure as login_err:
            raise ValueError(f"Login error occurred: {login_err}")
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")


class SendDiscordEmbedBlock(Block):
    class Input(BlockSchema):
        credentials: DiscordCredentials = DiscordCredentialsField()
        channel_identifier: str = SchemaField(
            description="Channel ID or channel name to send the embed to"
        )
        server_name: str = SchemaField(
            description="Server name (only needed if using channel name)",
            advanced=True,
            default="",
        )
        title: str = SchemaField(description="The title of the embed", default="")
        description: str = SchemaField(
            description="The main content/description of the embed", default=""
        )
        color: int = SchemaField(
            description="Embed color as integer (e.g., 0x00ff00 for green)",
            advanced=True,
            default=0x5865F2,  # Discord blurple
        )
        thumbnail_url: str = SchemaField(
            description="URL for the thumbnail image", advanced=True, default=""
        )
        image_url: str = SchemaField(
            description="URL for the main embed image", advanced=True, default=""
        )
        author_name: str = SchemaField(
            description="Author name to display", advanced=True, default=""
        )
        footer_text: str = SchemaField(
            description="Footer text", advanced=True, default=""
        )
        fields: list[dict[str, Any]] = SchemaField(
            description="List of field dictionaries with 'name', 'value', and optional 'inline' keys",
            advanced=True,
            default=[],
        )

    class Output(BlockSchema):
        status: str = SchemaField(description="Operation status")
        message_id: str = SchemaField(description="ID of the sent embed message")

    def __init__(self):
        super().__init__(
            id="c76293f4-9ae8-454d-a029-0a3f8c5bc499",
            input_schema=SendDiscordEmbedBlock.Input,
            output_schema=SendDiscordEmbedBlock.Output,
            description="Sends a rich embed message to a Discord channel.",
            categories={BlockCategory.SOCIAL},
            test_input={
                "channel_identifier": "general",
                "title": "Test Embed",
                "description": "This is a test embed message",
                "color": 0x00FF00,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("status", "Embed sent successfully"),
                ("message_id", "123456789012345678"),
            ],
            test_mock={
                "send_embed": lambda *args, **kwargs: {
                    "status": "Embed sent successfully",
                    "message_id": "123456789012345678",
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    async def send_embed(
        self,
        token: str,
        channel_identifier: str,
        server_name: str | None,
        embed_data: dict,
    ) -> dict:
        intents = discord.Intents.default()
        intents.guilds = True
        client = discord.Client(intents=intents)

        result = {}

        @client.event
        async def on_ready():
            channel = None

            # Try to parse as channel ID first
            try:
                channel_id = int(channel_identifier)
                channel = client.get_channel(channel_id)
            except ValueError:
                # Not an ID, treat as channel name
                for guild in client.guilds:
                    if server_name and guild.name != server_name:
                        continue
                    for ch in guild.text_channels:
                        if ch.name == channel_identifier:
                            channel = ch
                            break
                    if channel:
                        break

            if not channel:
                result["status"] = f"Channel not found: {channel_identifier}"
                await client.close()
                return

            # Build the embed
            embed = discord.Embed(
                title=embed_data.get("title") or None,
                description=embed_data.get("description") or None,
                color=embed_data.get("color", 0x5865F2),
            )

            if embed_data.get("thumbnail_url"):
                embed.set_thumbnail(url=embed_data["thumbnail_url"])

            if embed_data.get("image_url"):
                embed.set_image(url=embed_data["image_url"])

            if embed_data.get("author_name"):
                embed.set_author(name=embed_data["author_name"])

            if embed_data.get("footer_text"):
                embed.set_footer(text=embed_data["footer_text"])

            # Add fields
            for field in embed_data.get("fields", []):
                if isinstance(field, dict) and "name" in field and "value" in field:
                    embed.add_field(
                        name=field["name"],
                        value=field["value"],
                        inline=field.get("inline", True),
                    )

            try:
                # Type check - ensure it's a text channel that can send messages
                if not hasattr(channel, "send"):
                    result["status"] = (
                        f"Channel {channel_identifier} cannot receive messages (not a text channel)"
                    )
                    await client.close()
                    return

                message = await channel.send(embed=embed)  # type: ignore
                result["status"] = "Embed sent successfully"
                result["message_id"] = str(message.id)
            except Exception as e:
                result["status"] = f"Error sending embed: {str(e)}"
            finally:
                await client.close()

        await client.start(token)
        return result

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            embed_data = {
                "title": input_data.title,
                "description": input_data.description,
                "color": input_data.color,
                "thumbnail_url": input_data.thumbnail_url,
                "image_url": input_data.image_url,
                "author_name": input_data.author_name,
                "footer_text": input_data.footer_text,
                "fields": input_data.fields,
            }

            result = await self.send_embed(
                token=credentials.api_key.get_secret_value(),
                channel_identifier=input_data.channel_identifier,
                server_name=input_data.server_name or None,
                embed_data=embed_data,
            )

            yield "status", result.get("status", "Unknown error")
            if "message_id" in result:
                yield "message_id", result["message_id"]

        except discord.errors.LoginFailure as login_err:
            raise ValueError(f"Login error occurred: {login_err}")
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")


class SendDiscordFileBlock(Block):
    class Input(BlockSchema):
        credentials: DiscordCredentials = DiscordCredentialsField()
        channel_identifier: str = SchemaField(
            description="Channel ID or channel name to send the file to"
        )
        server_name: str = SchemaField(
            description="Server name (only needed if using channel name)",
            advanced=True,
            default="",
        )
        file: MediaFileType = SchemaField(
            description="The file to send (URL, data URI, or local path). Supports images, videos, documents, etc."
        )
        filename: str = SchemaField(
            description="Name of the file when sent (e.g., 'report.pdf', 'image.png')",
            default="",
        )
        message_content: str = SchemaField(
            description="Optional message to send with the file", default=""
        )

    class Output(BlockSchema):
        status: str = SchemaField(description="Operation status")
        message_id: str = SchemaField(description="ID of the sent message")

    def __init__(self):
        super().__init__(
            id="b1628cf2-4622-49bf-80cf-10e55826e247",
            input_schema=SendDiscordFileBlock.Input,
            output_schema=SendDiscordFileBlock.Output,
            description="Sends a file attachment to a Discord channel.",
            categories={BlockCategory.SOCIAL},
            test_input={
                "channel_identifier": "general",
                "file": "data:text/plain;base64,VGVzdCBmaWxlIGNvbnRlbnQ=",
                "filename": "test.txt",
                "message_content": "Here's the file!",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("status", "File sent successfully"),
                ("message_id", "123456789012345678"),
            ],
            test_mock={
                "send_file": lambda *args, **kwargs: {
                    "status": "File sent successfully",
                    "message_id": "123456789012345678",
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    async def send_file(
        self,
        token: str,
        channel_identifier: str,
        server_name: str | None,
        file: MediaFileType,
        filename: str,
        message_content: str,
        graph_exec_id: str,
        user_id: str,
    ) -> dict:
        intents = discord.Intents.default()
        intents.guilds = True
        client = discord.Client(intents=intents)

        result = {}

        @client.event
        async def on_ready():
            channel = None

            # Try to parse as channel ID first
            try:
                channel_id = int(channel_identifier)
                channel = client.get_channel(channel_id)
            except ValueError:
                # Not an ID, treat as channel name
                for guild in client.guilds:
                    if server_name and guild.name != server_name:
                        continue
                    for ch in guild.text_channels:
                        if ch.name == channel_identifier:
                            channel = ch
                            break
                    if channel:
                        break

            if not channel:
                result["status"] = f"Channel not found: {channel_identifier}"
                await client.close()
                return

            try:
                # Handle MediaFileType - could be data URI, URL, or local path
                file_bytes = None
                detected_filename = filename

                if file.startswith("data:"):
                    # Data URI - extract the base64 content
                    header, encoded = file.split(",", 1)
                    file_bytes = base64.b64decode(encoded)

                    # Try to get MIME type and suggest filename if not provided
                    if not filename and ";" in header:
                        mime_match = header.split(":")[1].split(";")[0]
                        ext = mimetypes.guess_extension(mime_match) or ".bin"
                        detected_filename = f"file{ext}"

                elif file.startswith(("http://", "https://")):
                    # URL - download the file
                    response = await Requests().get(file)
                    file_bytes = response.content

                    # Try to get filename from URL if not provided
                    if not filename:
                        from urllib.parse import urlparse

                        path = urlparse(file).path
                        detected_filename = Path(path).name or "download"
                else:
                    # Local file path - read from stored media file
                    # This would be a path from a previous block's output
                    stored_file = await store_media_file(
                        graph_exec_id=graph_exec_id,
                        file=file,
                        user_id=user_id,
                        return_content=True,  # Get as data URI
                    )
                    # Now process as data URI
                    header, encoded = stored_file.split(",", 1)
                    file_bytes = base64.b64decode(encoded)

                    if not filename:
                        detected_filename = Path(file).name or "file"

                if not file_bytes:
                    result["status"] = "Error: Could not read file content"
                    await client.close()
                    return

                # Create Discord file object
                discord_file = discord.File(
                    io.BytesIO(file_bytes), filename=detected_filename or "file"
                )

                # Type check - ensure it's a text channel that can send messages
                if not hasattr(channel, "send"):
                    result["status"] = (
                        f"Channel {channel_identifier} cannot receive messages (not a text channel)"
                    )
                    await client.close()
                    return

                # Send the file
                message = await channel.send(  # type: ignore
                    content=message_content if message_content else None,
                    file=discord_file,
                )
                result["status"] = "File sent successfully"
                result["message_id"] = str(message.id)
            except Exception as e:
                result["status"] = f"Error sending file: {str(e)}"
            finally:
                await client.close()

        await client.start(token)
        return result

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        graph_exec_id: str,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        try:
            result = await self.send_file(
                token=credentials.api_key.get_secret_value(),
                channel_identifier=input_data.channel_identifier,
                server_name=input_data.server_name or None,
                file=input_data.file,
                filename=input_data.filename,
                message_content=input_data.message_content,
                graph_exec_id=graph_exec_id,
                user_id=user_id,
            )

            yield "status", result.get("status", "Unknown error")
            if "message_id" in result:
                yield "message_id", result["message_id"]

        except discord.errors.LoginFailure as login_err:
            raise ValueError(f"Login error occurred: {login_err}")
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")


class ReplyToDiscordMessageBlock(Block):
    class Input(BlockSchema):
        credentials: DiscordCredentials = DiscordCredentialsField()
        channel_id: str = SchemaField(
            description="The channel ID where the message to reply to is located"
        )
        message_id: str = SchemaField(description="The ID of the message to reply to")
        reply_content: str = SchemaField(description="The content of the reply")
        mention_author: bool = SchemaField(
            description="Whether to mention the original message author", default=True
        )

    class Output(BlockSchema):
        status: str = SchemaField(description="Operation status")
        reply_id: str = SchemaField(description="ID of the reply message")

    def __init__(self):
        super().__init__(
            id="7226cb99-6e7b-4672-b6b2-acec95336eec",
            input_schema=ReplyToDiscordMessageBlock.Input,
            output_schema=ReplyToDiscordMessageBlock.Output,
            description="Replies to a specific Discord message.",
            categories={BlockCategory.SOCIAL},
            test_input={
                "channel_id": "123456789012345678",
                "message_id": "987654321098765432",
                "reply_content": "This is a reply!",
                "mention_author": True,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("status", "Reply sent successfully"),
                ("reply_id", "111222333444555666"),
            ],
            test_mock={
                "send_reply": lambda *args, **kwargs: {
                    "status": "Reply sent successfully",
                    "reply_id": "111222333444555666",
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    async def send_reply(
        self,
        token: str,
        channel_id: str,
        message_id: str,
        reply_content: str,
        mention_author: bool,
    ) -> dict:
        intents = discord.Intents.default()
        intents.guilds = True
        intents.message_content = True
        client = discord.Client(intents=intents)

        result = {}

        @client.event
        async def on_ready():
            try:
                channel = client.get_channel(int(channel_id))
                if not channel:
                    channel = await client.fetch_channel(int(channel_id))

                if not channel:
                    result["status"] = f"Channel with ID {channel_id} not found"
                    await client.close()
                    return

                # Type check - ensure it's a text channel that can fetch messages
                if not hasattr(channel, "fetch_message"):
                    result["status"] = (
                        f"Channel {channel_id} cannot fetch messages (not a text channel)"
                    )
                    await client.close()
                    return

                # Fetch the message to reply to
                try:
                    message = await channel.fetch_message(int(message_id))  # type: ignore
                except discord.errors.NotFound:
                    result["status"] = f"Message with ID {message_id} not found"
                    await client.close()
                    return

                # Send the reply
                reply = await message.reply(
                    content=reply_content, mention_author=mention_author
                )
                result["status"] = "Reply sent successfully"
                result["reply_id"] = str(reply.id)

            except ValueError as e:
                result["status"] = f"Invalid ID format: {str(e)}"
            except Exception as e:
                result["status"] = f"Error sending reply: {str(e)}"
            finally:
                await client.close()

        await client.start(token)
        return result

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = await self.send_reply(
                token=credentials.api_key.get_secret_value(),
                channel_id=input_data.channel_id,
                message_id=input_data.message_id,
                reply_content=input_data.reply_content,
                mention_author=input_data.mention_author,
            )

            yield "status", result.get("status", "Unknown error")
            if "reply_id" in result:
                yield "reply_id", result["reply_id"]

        except discord.errors.LoginFailure as login_err:
            raise ValueError(f"Login error occurred: {login_err}")
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")


class DiscordUserInfoBlock(Block):
    class Input(BlockSchema):
        credentials: DiscordCredentials = DiscordCredentialsField()
        user_id: str = SchemaField(
            description="The Discord user ID to get information about"
        )

    class Output(BlockSchema):
        user_id: str = SchemaField(
            description="The user's ID (passed through for chaining)"
        )
        username: str = SchemaField(description="The user's username")
        display_name: str = SchemaField(description="The user's display name")
        discriminator: str = SchemaField(
            description="The user's discriminator (if applicable)"
        )
        avatar_url: str = SchemaField(description="URL to the user's avatar")
        is_bot: bool = SchemaField(description="Whether the user is a bot")
        created_at: str = SchemaField(description="When the account was created")

    def __init__(self):
        super().__init__(
            id="9aeed32a-6ebf-49b8-a0a3-e2e509d86120",
            input_schema=DiscordUserInfoBlock.Input,
            output_schema=DiscordUserInfoBlock.Output,
            description="Gets information about a Discord user by their ID.",
            categories={BlockCategory.SOCIAL},
            test_input={
                "user_id": "123456789012345678",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("user_id", "123456789012345678"),
                ("username", "testuser"),
                ("display_name", "Test User"),
                ("discriminator", "0"),
                (
                    "avatar_url",
                    "https://cdn.discordapp.com/avatars/123456789012345678/avatar.png",
                ),
                ("is_bot", False),
                ("created_at", "2020-01-01T00:00:00"),
            ],
            test_mock={
                "get_user_info": lambda token, user_id: {
                    "user_id": "123456789012345678",
                    "username": "testuser",
                    "display_name": "Test User",
                    "discriminator": "0",
                    "avatar_url": "https://cdn.discordapp.com/avatars/123456789012345678/avatar.png",
                    "is_bot": False,
                    "created_at": "2020-01-01T00:00:00",
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    async def get_user_info(self, token: str, user_id: str) -> dict:
        intents = discord.Intents.default()
        client = discord.Client(intents=intents)

        result = {}

        @client.event
        async def on_ready():
            try:
                user = await client.fetch_user(int(user_id))

                result["user_id"] = str(user.id)  # Pass through the user ID
                result["username"] = user.name
                result["display_name"] = user.display_name or user.name
                result["discriminator"] = user.discriminator
                result["avatar_url"] = (
                    str(user.avatar.url)
                    if user.avatar
                    else str(user.default_avatar.url)
                )
                result["is_bot"] = user.bot
                result["created_at"] = user.created_at.isoformat()

            except discord.errors.NotFound:
                result["error"] = f"User with ID {user_id} not found"
            except ValueError:
                result["error"] = f"Invalid user ID format: {user_id}"
            except Exception as e:
                result["error"] = f"Error fetching user info: {str(e)}"
            finally:
                await client.close()

        await client.start(token)
        return result

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = await self.get_user_info(
                token=credentials.api_key.get_secret_value(), user_id=input_data.user_id
            )

            if "error" in result:
                raise ValueError(result["error"])

            yield "user_id", result["user_id"]
            yield "username", result["username"]
            yield "display_name", result["display_name"]
            yield "discriminator", result["discriminator"]
            yield "avatar_url", result["avatar_url"]
            yield "is_bot", result["is_bot"]
            yield "created_at", result["created_at"]

        except discord.errors.LoginFailure as login_err:
            raise ValueError(f"Login error occurred: {login_err}")
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")


class DiscordChannelInfoBlock(Block):
    class Input(BlockSchema):
        credentials: DiscordCredentials = DiscordCredentialsField()
        channel_identifier: str = SchemaField(
            description="Channel name or channel ID to look up"
        )
        server_name: str = SchemaField(
            description="Server name (optional, helps narrow down search)",
            advanced=True,
            default="",
        )

    class Output(BlockSchema):
        channel_id: str = SchemaField(description="The channel's ID")
        channel_name: str = SchemaField(description="The channel's name")
        server_id: str = SchemaField(description="The server's ID")
        server_name: str = SchemaField(description="The server's name")
        channel_type: str = SchemaField(
            description="Type of channel (text, voice, etc)"
        )

    def __init__(self):
        super().__init__(
            id="592f815e-35c3-4fed-96cd-a69966b45c8f",
            input_schema=DiscordChannelInfoBlock.Input,
            output_schema=DiscordChannelInfoBlock.Output,
            description="Resolves Discord channel names to IDs and vice versa.",
            categories={BlockCategory.SOCIAL},
            test_input={
                "channel_identifier": "general",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("channel_id", "123456789012345678"),
                ("channel_name", "general"),
                ("server_id", "987654321098765432"),
                ("server_name", "Test Server"),
                ("channel_type", "text"),
            ],
            test_mock={
                "get_channel_info": lambda token, channel_identifier, server_name: {
                    "channel_id": "123456789012345678",
                    "channel_name": "general",
                    "server_id": "987654321098765432",
                    "server_name": "Test Server",
                    "channel_type": "text",
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    async def get_channel_info(
        self, token: str, channel_identifier: str, server_name: str | None
    ) -> dict:
        intents = discord.Intents.default()
        intents.guilds = True
        client = discord.Client(intents=intents)

        result = {}

        @client.event
        async def on_ready():
            # Try to parse as channel ID first
            channel = None
            try:
                channel_id = int(channel_identifier)
                channel = client.get_channel(channel_id)
                if channel:
                    result["channel_id"] = str(channel.id)
                    # Private channels may not have a name attribute
                    result["channel_name"] = getattr(channel, "name", "Private Channel")
                    # Check if channel has guild (not private)
                    if hasattr(channel, "guild"):
                        guild = getattr(channel, "guild", None)
                        if guild:
                            result["server_id"] = str(guild.id)
                            result["server_name"] = guild.name
                        else:
                            result["server_id"] = ""
                            result["server_name"] = "Direct Message"
                    else:
                        result["server_id"] = ""
                        result["server_name"] = "Direct Message"
                    # Get channel type safely
                    result["channel_type"] = str(getattr(channel, "type", "unknown"))
                    await client.close()
                    return
            except ValueError:
                # Not an ID, treat as channel name
                for guild in client.guilds:
                    if server_name and guild.name != server_name:
                        continue
                    for ch in guild.channels:
                        if ch.name == channel_identifier:
                            result["channel_id"] = str(ch.id)
                            result["channel_name"] = ch.name
                            result["server_id"] = str(guild.id)
                            result["server_name"] = guild.name
                            result["channel_type"] = str(ch.type)
                            await client.close()
                            return

            result["error"] = f"Channel not found: {channel_identifier}"
            await client.close()

        await client.start(token)
        return result

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = await self.get_channel_info(
                token=credentials.api_key.get_secret_value(),
                channel_identifier=input_data.channel_identifier,
                server_name=input_data.server_name or None,
            )

            if "error" in result:
                raise ValueError(result["error"])

            yield "channel_id", result["channel_id"]
            yield "channel_name", result["channel_name"]
            yield "server_id", result["server_id"]
            yield "server_name", result["server_name"]
            yield "channel_type", result["channel_type"]

        except discord.errors.LoginFailure as login_err:
            raise ValueError(f"Login error occurred: {login_err}")
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")
