from urllib.parse import quote

from pydantic import model_validator

from backend.blocks._base import BlockOutput, BlockSchemaInput, BlockSchemaOutput
from backend.data.execution import ExecutionContext
from backend.data.model import APIKeyCredentials, SchemaField
from backend.util.type import MediaFileType

from ._api import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    Slant3DCredentialsField,
    Slant3DCredentialsInput,
)
from .base import Slant3DBlockBase


class Slant3DSlicerBlock(Slant3DBlockBase):
    class Input(BlockSchemaInput):
        credentials: Slant3DCredentialsInput = Slant3DCredentialsField()
        file_url: MediaFileType = SchemaField(
            default=MediaFileType(""),
            description="STL file URL, workspace file, or data URI; ignored when file_id is set",
        )
        file_id: str = SchemaField(
            default="", description="Previously uploaded Slant3D public file service ID"
        )
        platform_id: str = SchemaField(
            default="",
            description="Platform ID for uploads; may be omitted with one enabled platform",
        )
        filament_id: str = SchemaField(
            default="",
            description="Filament public ID; defaults to Slant3D's PLA Black",
        )
        quantity: int = SchemaField(
            default=1, ge=1, description="Number of prints to estimate"
        )

        @model_validator(mode="after")
        def require_file(self):
            if not self.file_id and not self.file_url:
                raise ValueError("Provide file_id or file_url")
            return self

    class Output(BlockSchemaOutput):
        message: str = SchemaField(description="Response message")
        price: float = SchemaField(
            description="Estimated printing price for the requested quantity in USD"
        )
        file_id: str = SchemaField(
            description="Slant3D public file service ID for order items"
        )

    def __init__(self):
        super().__init__(
            id="f8a12c8d-3e4b-4d5f-b6a7-8c9d0e1f2g3h",
            description=(
                "Get a live 3D printing quote for physical parts from STL URLs or attached workspace files. "
                "Use this to price printable parts and complete project part sets before ordering from Slant3D, "
                "which manufactures and ships the parts. Binary STL URLs are accepted directly; no browser upload is needed. "
                "For complete projects, first read the assembly bill of materials: include repeated parts across subassemblies, "
                "select a design variant for alternative parts while keeping the required number of copies, "
                "and exclude reference assembly meshes. "
                "One copy of each STL is not necessarily a complete set. "
                "Quote each required file with its full assembly quantity and sum the returned prices, which already include quantity. "
                "Before reporting a complete quote, reconcile all quoted files and quantities against the parts list. "
                "Printing-only quotes need no shipping address or payment method. "
                "Returns a reusable file_id for ordering; this block does not place or charge an order."
            ),
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "file_id": "22222222-2222-4222-8222-222222222222",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("message", "File Price Estimated"),
                ("price", 8.23),
                ("file_id", "22222222-2222-4222-8222-222222222222"),
            ],
            test_mock={
                "_make_request": lambda *args, **kwargs: {
                    "message": "File Price Estimated",
                    "data": {"total": 8.23},
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        file_id = input_data.file_id
        if not file_id:
            platform_id = await self._resolve_platform_id(
                input_data.platform_id, api_key
            )
            file_id = await self._upload_file(
                input_data.file_url,
                platform_id,
                api_key,
                execution_context=execution_context,
            )
        options: dict[str, str | int] = {"quantity": input_data.quantity}
        if input_data.filament_id:
            options["filamentId"] = input_data.filament_id
        result = await self._make_request(
            "POST",
            f"files/{quote(file_id, safe='')}/estimate",
            api_key,
            json={"options": options},
        )
        quoted_quantity = result["data"].get("quantity", 1)
        if quoted_quantity != input_data.quantity:
            raise ValueError(
                f"Slant3D returned a price for {quoted_quantity} print(s), "
                f"but {input_data.quantity} were requested. "
                "No valid total was returned for the requested quantity."
            )
        total = result["data"].get("total")
        if total is None:
            raise ValueError("Slant3D did not return a printing total")
        price = float(total)
        yield "message", result["message"]
        yield "price", price
        yield "file_id", file_id
