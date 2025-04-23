from backend.data.block import BlockOutput, BlockSchema
from backend.data.model import APIKeyCredentials, SchemaField

from ._api import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    Slant3DCredentialsField,
    Slant3DCredentialsInput,
)
from .base import Slant3DBlockBase


class Slant3DSlicerBlock(Slant3DBlockBase):
    """Block for slicing 3D model files"""

    class Input(BlockSchema):
        credentials: Slant3DCredentialsInput = Slant3DCredentialsField()
        file_url: str = SchemaField(
            description="URL of the 3D model file to slice (STL)"
        )

    class Output(BlockSchema):
        message: str = SchemaField(description="Response message")
        price: float = SchemaField(description="Calculated price for printing")
        error: str = SchemaField(description="Error message if slicing failed")

    def __init__(self):
        super().__init__(
            id="f8a12c8d-3e4b-4d5f-b6a7-8c9d0e1f2g3h",
            description="Slice a 3D model file and get pricing information",
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "file_url": "https://example.com/model.stl",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("message", "Slicing successful"), ("price", 8.23)],
            test_mock={
                "_make_request": lambda *args, **kwargs: {
                    "message": "Slicing successful",
                    "data": {"price": 8.23},
                }
            },
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = self._make_request(
                "POST",
                "slicer",
                credentials.api_key.get_secret_value(),
                json={"fileURL": input_data.file_url},
            )
            yield "message", result["message"]
            yield "price", result["data"]["price"]
        except Exception as e:
            yield "error", str(e)
            raise
