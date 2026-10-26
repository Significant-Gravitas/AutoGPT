import asyncio

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField
from backend.notifications.owner_email_body import (
    BodyFormat,
    InlineImage,
    image_sources,
)
from backend.notifications.owner_email_images import fetch_images
from backend.util.clients import get_notification_manager_client


class SendEmailToMeBlock(Block):
    class Input(BlockSchemaInput):
        subject: str = SchemaField(
            description="Subject of the email", placeholder="Your weekly summary"
        )
        body: str = SchemaField(
            description=(
                "Body of the email, written in the chosen format. Images "
                "(![alt](workspace://<file_id>) or an https:// URL in Markdown, <img> "
                "in HTML) are attached to the email."
            ),
            placeholder="Here is what I found this week...",
        )
        body_format: BodyFormat = SchemaField(
            description=(
                "How the body is written: Markdown, plain text, or HTML (scripts, "
                "forms and styles are removed)"
            ),
            default=BodyFormat.MARKDOWN,
        )
        link: str = SchemaField(
            description="Optional web link to show as a button, e.g. to the full result",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        status: str = SchemaField(description="Status of the email sending operation")

    def __init__(self):
        super().__init__(
            id="38997608-a57a-4e11-971c-b83a5332e80b",
            description=(
                "Email yourself: sends a message to your own inbox, at your account's "
                "email address, from the platform, with no email account to sign in "
                "to. Use it for "
                '"email me" or "send it to myself": a result, a summary or a '
                "recurring digest. It can't send to anyone else; to email other "
                "people, use Gmail or SMTP."
            ),
            categories={BlockCategory.OUTPUT, BlockCategory.COMMUNICATION},
            input_schema=SendEmailToMeBlock.Input,
            output_schema=SendEmailToMeBlock.Output,
            test_input={"subject": "Test Email", "body": "This is a test email."},
            test_output=[("status", "Email sent to your account address")],
            test_mock={"send_to_owner": lambda *args, **kwargs: None},
            # The only recipient is the account owner: a self-DM, not a send to
            # someone outside the platform.
            capability_kind="service",
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    async def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        try:
            sources = await asyncio.to_thread(
                image_sources, input_data.body, input_data.body_format
            )
            images = await fetch_images(sources, execution_context)
            await self.send_to_owner(
                user_id,
                input_data.subject,
                input_data.body,
                input_data.link or None,
                input_data.body_format,
                images,
            )
        except (ValueError, RuntimeError, TimeoutError) as e:
            yield "error", str(e)
            return
        yield "status", "Email sent to your account address"

    @staticmethod
    async def send_to_owner(
        user_id: str,
        subject: str,
        body: str,
        link: str | None,
        body_format: BodyFormat,
        images: list[InlineImage],
    ) -> None:
        await get_notification_manager_client().send_owner_email(
            user_id, subject, body, link, body_format, images
        )
