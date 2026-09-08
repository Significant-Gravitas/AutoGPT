from backend.blocks._base import BlockOutput, BlockSchemaInput, BlockSchemaOutput
from backend.data.model import APIKeyCredentials, SchemaField

from ._api import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    Filament,
    Profile,
    Slant3DCredentialsField,
    Slant3DCredentialsInput,
)
from .base import Slant3DBlockBase

TEST_FILAMENT = {
    "publicId": "33333333-3333-4333-8333-333333333333",
    "name": "PLA BLACK",
    "provider": "Tangled",
    "profile": "PLA",
    "color": "black",
    "hexValue": "#000000",
}


class Slant3DFilamentBlock(Slant3DBlockBase):
    class Input(BlockSchemaInput):
        credentials: Slant3DCredentialsInput = Slant3DCredentialsField()
        profiles: list[Profile] = SchemaField(
            default_factory=list, description="Filter materials; empty returns all"
        )
        colors: list[str] = SchemaField(
            default_factory=list, description="Filter color names; empty returns all"
        )

    class Output(BlockSchemaOutput):
        filaments: list[Filament] = SchemaField(
            description="Available filaments; use publicId as filament_id"
        )

    def __init__(self):
        super().__init__(
            id="7cc416f4-f305-4606-9b3b-452b8a81031c",
            description="Get available filaments, their public IDs, and material and color details",
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "filaments",
                    [
                        {
                            **TEST_FILAMENT,
                            "filament": "PLA BLACK",
                            "hexColor": "000000",
                            "colorTag": "black",
                        }
                    ],
                )
            ],
            test_mock={
                "_make_request": lambda *args, **kwargs: {"data": [TEST_FILAMENT]}
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        params = {}
        if input_data.profiles:
            params["profile"] = ",".join(
                profile.value for profile in input_data.profiles
            )
        if input_data.colors:
            params["color"] = ",".join(input_data.colors)
        result = await self._make_request(
            "GET",
            "filaments",
            credentials.api_key.get_secret_value(),
            params=params,
        )
        yield "filaments", [
            {
                **filament,
                "filament": filament["name"],
                "hexColor": filament["hexValue"].removeprefix("#"),
                "colorTag": (
                    filament["color"]
                    if filament["profile"] == "PLA"
                    else f"{filament['profile'].lower()}{filament['color'].capitalize()}"
                ),
            }
            for filament in result["data"]
        ]
