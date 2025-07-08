"""
Airtable attachment blocks.
"""

import base64
from typing import Union

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    Requests,
    SchemaField,
)

from ._config import airtable


class AirtableUploadAttachmentBlock(Block):
    """
    Uploads a file to Airtable for use as an attachment.

    Files can be uploaded directly (up to 5MB) or via URL.
    The returned attachment ID can be used when creating or updating records.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID", default="")
        filename: str = SchemaField(description="Name of the file")
        file: Union[bytes, str] = SchemaField(
            description="File content (binary data or base64 string)"
        )
        content_type: str = SchemaField(
            description="MIME type of the file", default="application/octet-stream"
        )

    class Output(BlockSchema):
        attachment: dict = SchemaField(
            description="Attachment object with id, url, size, and type"
        )
        attachment_id: str = SchemaField(description="ID of the uploaded attachment")
        url: str = SchemaField(description="URL of the uploaded attachment")
        size: int = SchemaField(description="Size of the file in bytes")

    def __init__(self):
        super().__init__(
            id="962e801b-5a6f-4c56-a929-83e816343a41",
            description="Upload a file to Airtable for use as an attachment",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Convert file to base64 if it's bytes
        if isinstance(input_data.file, bytes):
            file_data = base64.b64encode(input_data.file).decode("utf-8")
        else:
            # Assume it's already base64 encoded
            file_data = input_data.file

        # Check file size (5MB limit)
        file_bytes = base64.b64decode(file_data)
        if len(file_bytes) > 5 * 1024 * 1024:
            raise ValueError(
                "File size exceeds 5MB limit. Use URL upload for larger files."
            )

        # Upload the attachment
        response = await Requests().post(
            f"https://api.airtable.com/v0/bases/{input_data.base_id}/attachments/upload",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "content": file_data,
                "filename": input_data.filename,
                "type": input_data.content_type,
            },
        )

        attachment_data = response.json()

        yield "attachment", attachment_data
        yield "attachment_id", attachment_data.get("id", "")
        yield "url", attachment_data.get("url", "")
        yield "size", attachment_data.get("size", 0)
