from typing import Any

from backend.blocks._base import BlockSchemaInput, BlockSchemaOutput
from backend.data.model import SchemaField
from backend.util.file import MediaFileType

from ._client import MAX_IMAGES
from ._config import NaceCredentialsField, NaceCredentialsInput

CALIBRATION = (
    "Levels define the scale: describe concrete evidence for each tier so a strong "
    "case does not automatically reach the top. For hiring, distinguish terrible, bad, "
    "okay, excellent, brilliant, and one-in-a-thousand by what qualifies."
)


class DrexInput(BlockSchemaInput):
    credentials: NaceCredentialsInput = NaceCredentialsField()
    state: Any = SchemaField(
        default=None,
        advanced=False,
        description=(
            "Text or JSON context shared by every question in this stateless call. "
            "JSON is compactly serialized; oversized context is truncated with a note."
        ),
    )
    images: list[MediaFileType] = SchemaField(
        default=[],
        advanced=False,
        max_length=MAX_IMAGES,
        description=(
            f"Up to {MAX_IMAGES} PNG, JPEG or WebP images, each a data URL, an "
            "http(s) URL or a workspace file. Refer to them in the question as "
            "'Image 1', 'Image 2' in list order."
        ),
    )


class DrexOutput(BlockSchemaOutput):
    request: str = SchemaField(
        description=(
            "Request JSON sent to nace; inline image data is represented by "
            "its MIME type and size."
        )
    )
    response: str | None = SchemaField(
        description=(
            "Verbatim UTF-8 response body from nace, or null if no response arrived. "
            "Invalid UTF-8 is preserved as a lossless Base64 data URL with an error."
        )
    )
    latency_ms: float = SchemaField(description="Wall-clock API call duration in ms.")
    input_tokens: int | None = SchemaField(
        description="Input tokens reported by the API, or null if unavailable on failure."
    )
    output_tokens: int | None = SchemaField(
        description="Output tokens reported by the API, or null if unavailable on failure."
    )
    request_id: str = SchemaField(description="nace request ID.")
    truncated: bool = SchemaField(description="Whether the state text was truncated.")
    truncation_note: str = SchemaField(
        description="Truncation details, or an empty string."
    )
    error: str = SchemaField(
        default="", description="API failure details; emitted only when a call fails."
    )
