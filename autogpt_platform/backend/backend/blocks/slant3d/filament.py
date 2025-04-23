from typing import List

from backend.data.block import BlockOutput, BlockSchema
from backend.data.model import APIKeyCredentials, SchemaField

from ._api import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    Filament,
    Slant3DCredentialsField,
    Slant3DCredentialsInput,
)
from .base import Slant3DBlockBase


class Slant3DFilamentBlock(Slant3DBlockBase):
    """Block for retrieving available filaments"""

    class Input(BlockSchema):
        credentials: Slant3DCredentialsInput = Slant3DCredentialsField()

    class Output(BlockSchema):
        filaments: List[Filament] = SchemaField(
            description="List of available filaments"
        )
        error: str = SchemaField(description="Error message if request failed")

    def __init__(self):
        super().__init__(
            id="7cc416f4-f305-4606-9b3b-452b8a81031c",
            description="Get list of available filaments",
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "filaments",
                    [
                        {
                            "filament": "PLA BLACK",
                            "hexColor": "000000",
                            "colorTag": "black",
                            "profile": "PLA",
                        },
                        {
                            "filament": "PLA WHITE",
                            "hexColor": "ffffff",
                            "colorTag": "white",
                            "profile": "PLA",
                        },
                    ],
                )
            ],
            test_mock={
                "_make_request": lambda *args, **kwargs: {
                    "filaments": [
                        {
                            "filament": "PLA BLACK",
                            "hexColor": "000000",
                            "colorTag": "black",
                            "profile": "PLA",
                        },
                        {
                            "filament": "PLA WHITE",
                            "hexColor": "ffffff",
                            "colorTag": "white",
                            "profile": "PLA",
                        },
                    ]
                }
            },
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = self._make_request(
                "GET", "filament", credentials.api_key.get_secret_value()
            )
            yield "filaments", result["filaments"]
        except Exception as e:
            yield "error", str(e)
            raise
