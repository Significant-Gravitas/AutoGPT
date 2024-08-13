from typing import Literal

from forge.file_storage import FileStorage
from forge.file_storage.gcs import GCSFileStorage, GCSFileStorageConfiguration
from forge.file_storage.google_drive import (
    GoogleDriveFileStorage,
    GoogleDriveFileStorageConfiguration,
)

from autogpt_server.data.block import Block, BlockOutput, BlockSchema
from autogpt_server.data.model import SchemaField


class _GCSFileStorageConfig(GCSFileStorageConfiguration):
    provider: Literal["gcs"]


class _GoogleDriveFileStorageConfig(GoogleDriveFileStorageConfiguration):
    provider: Literal["google_drive"]


_FileStorageConfig = _GCSFileStorageConfig | _GoogleDriveFileStorageConfig


def _get_storage(config: _FileStorageConfig) -> FileStorage:
    if config.provider == "google_drive":
        return GoogleDriveFileStorage(config)
    if config.provider == "gcs":
        return GCSFileStorage(config)

    raise TypeError(f"Invalid storage configuration: {config}")


class ReadFileBlock(Block):
    class Input(BlockSchema):
        file_storage: _FileStorageConfig = SchemaField(
            description="Configuration for the file storage to use",
            json_schema_extra={"resource_type": "file_storage"},
        )
        path: str = SchemaField(
            description="The path of the file to read",
            placeholder="example.txt",
        )
        type: Literal["text", "bytes"] = SchemaField(
            description="The type of the file content",
            default="text",
        )

    class Output(BlockSchema):
        content: str | bytes = SchemaField(description="The content of the read file")
        length: int = SchemaField(
            description="The length/size of the file content (bytes)"
        )
        error: str = SchemaField(
            description="Any error message if the file can't be read"
        )

    def __init__(self):
        super().__init__(
            id="e58cdb7c-f2d2-42ea-8c79-d6eaabd7df3b",
            input_schema=ReadFileBlock.Input,
            output_schema=ReadFileBlock.Output,
        )

    def run(self, input_data: Input) -> BlockOutput:
        try:
            storage = _get_storage(input_data.file_storage)
            content = storage.read_file(input_data.path, input_data.type == "bytes")
            yield "content", content
            yield "length", len(content)
        except Exception as e:
            yield "error", str(e)
