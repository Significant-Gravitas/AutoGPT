import json
import logging
from enum import Enum
from io import BytesIO
from pathlib import Path

import aiofiles

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.file import (
    MediaFileType,
    get_exec_file_path,
    get_mime_type,
    store_media_file,
)
from backend.util.request import Requests

logger = logging.getLogger(name=__name__)


class HttpMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class SendWebRequestBlock(Block):
    class Input(BlockSchema):
        url: str = SchemaField(
            description="The URL to send the request to",
            placeholder="https://api.example.com",
        )
        method: HttpMethod = SchemaField(
            description="The HTTP method to use for the request",
            default=HttpMethod.POST,
        )
        headers: dict[str, str] = SchemaField(
            description="The headers to include in the request",
            default_factory=dict,
        )
        json_format: bool = SchemaField(
            title="JSON format",
            description="If true, send the body as JSON (unless files are also present).",
            default=True,
        )
        body: dict | None = SchemaField(
            description="Form/JSON body payload. If files are supplied, this must be a mapping of form‑fields.",
            default=None,
        )
        files_name: str = SchemaField(
            description="The name of the file field in the form data.",
            default="file",
        )
        files: list[MediaFileType] = SchemaField(
            description="Mapping of *form field name* → Image url / path / base64 url.",
            default_factory=list,
        )

    class Output(BlockSchema):
        response: object = SchemaField(description="The response from the server")
        client_error: object = SchemaField(description="Errors on 4xx status codes")
        server_error: object = SchemaField(description="Errors on 5xx status codes")
        error: str = SchemaField(description="Errors for all other exceptions")

    def __init__(self):
        super().__init__(
            id="6595ae1f-b924-42cb-9a41-551a0611c4b4",
            description="Make an HTTP request (JSON / form / multipart).",
            categories={BlockCategory.OUTPUT},
            input_schema=SendWebRequestBlock.Input,
            output_schema=SendWebRequestBlock.Output,
        )

    @staticmethod
    async def _prepare_files(
        graph_exec_id: str,
        files_name: str,
        files: list[MediaFileType],
    ) -> list[tuple[str, tuple[str, BytesIO, str]]]:
        """
        Prepare files for the request by storing them and reading their content.
        Returns a list of tuples in the format:
        (files_name, (filename, BytesIO, mime_type))
        """
        files_payload: list[tuple[str, tuple[str, BytesIO, str]]] = []

        for media in files:
            # Normalise to a list so we can repeat the same key
            rel_path = await store_media_file(
                graph_exec_id, media, return_content=False
            )
            abs_path = get_exec_file_path(graph_exec_id, rel_path)
            async with aiofiles.open(abs_path, "rb") as f:
                content = await f.read()
                handle = BytesIO(content)
                mime = get_mime_type(abs_path)
                files_payload.append((files_name, (Path(abs_path).name, handle, mime)))

        return files_payload

    async def run(
        self, input_data: Input, *, graph_exec_id: str, **kwargs
    ) -> BlockOutput:
        # ─── Parse/normalise body ────────────────────────────────────
        body = input_data.body
        if isinstance(body, str):
            try:
                # Validate JSON string length to prevent DoS attacks
                if len(body) > 10_000_000:  # 10MB limit
                    raise ValueError("JSON body too large")

                parsed_body = json.loads(body)

                # Validate that parsed JSON is safe (basic object/array/primitive types)
                if (
                    isinstance(parsed_body, (dict, list, str, int, float, bool))
                    or parsed_body is None
                ):
                    body = parsed_body
                else:
                    # Unexpected type, treat as plain text
                    input_data.json_format = False

            except (json.JSONDecodeError, ValueError):
                # Invalid JSON or too large – treat as form‑field value instead
                input_data.json_format = False

        # ─── Prepare files (if any) ──────────────────────────────────
        use_files = bool(input_data.files)
        files_payload: list[tuple[str, tuple[str, BytesIO, str]]] = []
        if use_files:
            files_payload = await self._prepare_files(
                graph_exec_id, input_data.files_name, input_data.files
            )

        # Enforce body format rules
        if use_files and input_data.json_format:
            raise ValueError(
                "json_format=True cannot be combined with file uploads; set json_format=False and put form fields in `body`."
            )

        # ─── Execute request ─────────────────────────────────────────
        response = await Requests().request(
            input_data.method.value,
            input_data.url,
            headers=input_data.headers,
            files=files_payload if use_files else None,
            # * If files → multipart ⇒ pass form‑fields via data=
            data=body if not input_data.json_format else None,
            # * Else, choose JSON vs url‑encoded based on flag
            json=body if (input_data.json_format and not use_files) else None,
        )

        # Decide how to parse the response
        if response.headers.get("content-type", "").startswith("application/json"):
            result = None if response.status == 204 else response.json()
        else:
            result = response.text()

        # Yield according to status code bucket
        if 200 <= response.status < 300:
            yield "response", result
        elif 400 <= response.status < 500:
            yield "client_error", result
        else:
            yield "server_error", result
