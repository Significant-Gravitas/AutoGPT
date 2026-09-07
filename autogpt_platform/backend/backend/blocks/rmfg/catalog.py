"""Blocks that read the RMFG catalogs: stock, finishes, colors and hardware."""

from typing import Optional

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import RMFGClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._inputs import credentials_field
from ._models import Finish, HardwareOption, Material, PowderCoatColor, TubeProfile
from ._testdata import (
    TEST_FINISH,
    TEST_HARDWARE_OPTION,
    TEST_MATERIAL,
    TEST_POWDER_COAT_COLOR,
    TEST_TUBE_PROFILE,
)
from ._types import HardwareKind, Process

CATEGORIES = {BlockCategory.HARDWARE, BlockCategory.DATA}


class RMFGListMaterialsBlock(Block):
    """List sheet-metal stock. Use a returned ``id`` as ``material_id``."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()

    class Output(BlockSchemaOutput):
        materials: list[Material] = SchemaField(
            description="Every sheet-metal material, across all pages"
        )
        material: Material = SchemaField(description="One material at a time")
        material_ids: list[str] = SchemaField(description="IDs in the same order")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="bd3afa7d-607e-4b0e-bd24-201c54fa0ad4",
            description="Lists the sheet-metal materials RMFG can cut and bend",
            categories=CATEGORIES,
            input_schema=RMFGListMaterialsBlock.Input,
            output_schema=RMFGListMaterialsBlock.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("materials", [TEST_MATERIAL]),
                ("material", TEST_MATERIAL),
                ("material_ids", [TEST_MATERIAL.id]),
            ],
            test_mock={"list_materials": lambda *args, **kwargs: [TEST_MATERIAL]},
        )

    @staticmethod
    async def list_materials(credentials: APIKeyCredentials) -> list[Material]:
        return await RMFGClient(credentials).list_materials()

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        materials = await self.list_materials(credentials)
        yield "materials", materials
        for material in materials:
            yield "material", material
        yield "material_ids", [material.id for material in materials]


class RMFGListTubeProfilesBlock(Block):
    """List tube-laser stock. Use a returned ``id`` as ``tube_profile_id``."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()

    class Output(BlockSchemaOutput):
        tube_profiles: list[TubeProfile] = SchemaField(
            description="Every tube profile, across all pages"
        )
        tube_profile: TubeProfile = SchemaField(description="One profile at a time")
        tube_profile_ids: list[str] = SchemaField(description="IDs in the same order")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="48356027-25ee-4711-ae2e-5204e5a5f717",
            description="Lists the tube stock profiles RMFG can laser-cut",
            categories=CATEGORIES,
            input_schema=RMFGListTubeProfilesBlock.Input,
            output_schema=RMFGListTubeProfilesBlock.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("tube_profiles", [TEST_TUBE_PROFILE]),
                ("tube_profile", TEST_TUBE_PROFILE),
                ("tube_profile_ids", [TEST_TUBE_PROFILE.id]),
            ],
            test_mock={
                "list_tube_profiles": lambda *args, **kwargs: [TEST_TUBE_PROFILE]
            },
        )

    @staticmethod
    async def list_tube_profiles(credentials: APIKeyCredentials) -> list[TubeProfile]:
        return await RMFGClient(credentials).list_tube_profiles()

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        profiles = await self.list_tube_profiles(credentials)
        yield "tube_profiles", profiles
        for profile in profiles:
            yield "tube_profile", profile
        yield "tube_profile_ids", [profile.id for profile in profiles]


class RMFGListFinishesBlock(Block):
    """List mechanical finishes. Use a returned ``id`` as ``finish_id``."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()
        process: Optional[Process] = SchemaField(
            description="Only finishes that apply to this process; empty for all.",
            default=None,
            advanced=False,
        )

    class Output(BlockSchemaOutput):
        finishes: list[Finish] = SchemaField(description="Matching finishes")
        finish: Finish = SchemaField(description="One finish at a time")
        finish_ids: list[str] = SchemaField(description="IDs in the same order")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="87dbab6f-26ac-4371-9aef-51ad30bc0584",
            description="Lists the finishes RMFG can apply to sheet or tube parts",
            categories=CATEGORIES,
            input_schema=RMFGListFinishesBlock.Input,
            output_schema=RMFGListFinishesBlock.Output,
            test_input={
                "process": Process.SHEET_METAL.value,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("finishes", [TEST_FINISH]),
                ("finish", TEST_FINISH),
                ("finish_ids", [TEST_FINISH.id]),
            ],
            test_mock={"list_finishes": lambda *args, **kwargs: [TEST_FINISH]},
        )

    @staticmethod
    async def list_finishes(
        credentials: APIKeyCredentials, process: Optional[Process]
    ) -> list[Finish]:
        return await RMFGClient(credentials).list_finishes(process)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        finishes = await self.list_finishes(credentials, input_data.process)
        yield "finishes", finishes
        for finish in finishes:
            yield "finish", finish
        yield "finish_ids", [finish.id for finish in finishes]


class RMFGListPowderCoatColorsBlock(Block):
    """List powder-coat colors. Use a returned ``id`` as ``powder_coat_color_id``."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()

    class Output(BlockSchemaOutput):
        colors: list[PowderCoatColor] = SchemaField(description="Every color")
        color: PowderCoatColor = SchemaField(description="One color at a time")
        color_ids: list[str] = SchemaField(description="IDs in the same order")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="e0eea253-ccdc-4ee8-bb16-68e444dad6fc",
            description="Lists the powder-coat colors RMFG offers",
            categories=CATEGORIES,
            input_schema=RMFGListPowderCoatColorsBlock.Input,
            output_schema=RMFGListPowderCoatColorsBlock.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("colors", [TEST_POWDER_COAT_COLOR]),
                ("color", TEST_POWDER_COAT_COLOR),
                ("color_ids", [TEST_POWDER_COAT_COLOR.id]),
            ],
            test_mock={"list_colors": lambda *args, **kwargs: [TEST_POWDER_COAT_COLOR]},
        )

    @staticmethod
    async def list_colors(credentials: APIKeyCredentials) -> list[PowderCoatColor]:
        return await RMFGClient(credentials).list_powder_coat_colors()

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        colors = await self.list_colors(credentials)
        yield "colors", colors
        for color in colors:
            yield "color", color
        yield "color_ids", [color.id for color in colors]


class RMFGListHardwareBlock(Block):
    """List taps, studs, nuts or standoffs for hole operations."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()
        kind: HardwareKind = SchemaField(
            description=(
                "Which catalog to read. Reference an entry's id as tap_id, "
                "stud_id, nut_id or standoff_id in a part configuration."
            ),
            default=HardwareKind.TAPS,
            advanced=False,
        )

    class Output(BlockSchemaOutput):
        options: list[HardwareOption] = SchemaField(description="Catalog entries")
        option: HardwareOption = SchemaField(description="One entry at a time")
        option_ids: list[str] = SchemaField(description="IDs in the same order")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="6496ec49-0a6d-4b74-9d2f-573bb09e735d",
            description="Lists the taps, studs, nuts or standoffs RMFG can install",
            categories=CATEGORIES,
            input_schema=RMFGListHardwareBlock.Input,
            output_schema=RMFGListHardwareBlock.Output,
            test_input={
                "kind": HardwareKind.TAPS.value,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("options", [TEST_HARDWARE_OPTION]),
                ("option", TEST_HARDWARE_OPTION),
                ("option_ids", [TEST_HARDWARE_OPTION.id]),
            ],
            test_mock={"list_hardware": lambda *args, **kwargs: [TEST_HARDWARE_OPTION]},
        )

    @staticmethod
    async def list_hardware(
        credentials: APIKeyCredentials, kind: HardwareKind
    ) -> list[HardwareOption]:
        return await RMFGClient(credentials).list_hardware(kind)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        options = await self.list_hardware(credentials, input_data.kind)
        yield "options", options
        for option in options:
            yield "option", option
        yield "option_ids", [option.id for option in options]
