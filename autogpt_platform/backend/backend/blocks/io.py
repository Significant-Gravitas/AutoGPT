import copy
from datetime import date, time
from typing import Any, Optional

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema, BlockType
from backend.data.model import SchemaField
from backend.util.file import store_media_file
from backend.util.mock import MockObject
from backend.util.settings import Config
from backend.util.text import TextFormatter
from backend.util.type import LongTextType, MediaFileType, ShortTextType

config = Config()


class AgentInputBlock(Block):
    """
    This block is used to provide input to the graph.

    It takes in a value, name, description, default values list and bool to limit selection to default values.

    It Outputs the value passed as input.
    """

    class Input(BlockSchema):
        name: str = SchemaField(description="The name of the input.")
        value: Any = SchemaField(
            description="The value to be passed as input.",
            default=None,
        )
        title: str | None = SchemaField(
            description="The title of the input.", default=None, advanced=True
        )
        description: str | None = SchemaField(
            description="The description of the input.",
            default=None,
            advanced=True,
        )
        placeholder_values: list = SchemaField(
            description="The placeholder values to be passed as input.",
            default_factory=list,
            advanced=True,
            hidden=True,
        )
        advanced: bool = SchemaField(
            description="Whether to show the input in the advanced section, if the field is not required.",
            default=False,
            advanced=True,
        )
        secret: bool = SchemaField(
            description="Whether the input should be treated as a secret.",
            default=False,
            advanced=True,
        )

        def generate_schema(self):
            schema = copy.deepcopy(self.get_field_schema("value"))
            if possible_values := self.placeholder_values:
                schema["enum"] = possible_values
            return schema

    class Output(BlockSchema):
        result: Any = SchemaField(description="The value passed as input.")

    def __init__(self, **kwargs):
        super().__init__(
            **{
                "id": "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",
                "description": "Base block for user inputs.",
                "input_schema": AgentInputBlock.Input,
                "output_schema": AgentInputBlock.Output,
                "test_input": [
                    {
                        "value": "Hello, World!",
                        "name": "input_1",
                        "description": "Example test input.",
                        "placeholder_values": [],
                    },
                    {
                        "value": "Hello, World!",
                        "name": "input_2",
                        "description": "Example test input with placeholders.",
                        "placeholder_values": ["Hello, World!"],
                    },
                ],
                "test_output": [
                    ("result", "Hello, World!"),
                    ("result", "Hello, World!"),
                ],
                "categories": {BlockCategory.INPUT, BlockCategory.BASIC},
                "block_type": BlockType.INPUT,
                "static_output": True,
                **kwargs,
            }
        )

    async def run(self, input_data: Input, *args, **kwargs) -> BlockOutput:
        if input_data.value is not None:
            yield "result", input_data.value


class AgentOutputBlock(Block):
    """
    Records the output of the graph for users to see.

    Behavior:
        If `format` is provided and the `value` is of a type that can be formatted,
        the block attempts to format the recorded_value using the `format`.
        If formatting fails or no `format` is provided, the raw `value` is output.
    """

    class Input(BlockSchema):
        value: Any = SchemaField(
            description="The value to be recorded as output.",
            default=None,
            advanced=False,
        )
        name: str = SchemaField(description="The name of the output.")
        title: str | None = SchemaField(
            description="The title of the output.",
            default=None,
            advanced=True,
        )
        description: str | None = SchemaField(
            description="The description of the output.",
            default=None,
            advanced=True,
        )
        format: str = SchemaField(
            description="The format string to be used to format the recorded_value. Use Jinja2 syntax.",
            default="",
            advanced=True,
        )
        escape_html: bool = SchemaField(
            default=False,
            advanced=True,
            description="Whether to escape special characters in the inserted values to be HTML-safe. Enable for HTML output, disable for plain text.",
        )
        advanced: bool = SchemaField(
            description="Whether to treat the output as advanced.",
            default=False,
            advanced=True,
        )
        secret: bool = SchemaField(
            description="Whether the output should be treated as a secret.",
            default=False,
            advanced=True,
        )

        def generate_schema(self):
            return self.get_field_schema("value")

    class Output(BlockSchema):
        output: Any = SchemaField(description="The value recorded as output.")
        name: Any = SchemaField(description="The name of the value recorded as output.")

    def __init__(self):
        super().__init__(
            id="363ae599-353e-4804-937e-b2ee3cef3da4",
            description="Stores the output of the graph for users to see.",
            input_schema=AgentOutputBlock.Input,
            output_schema=AgentOutputBlock.Output,
            test_input=[
                {
                    "value": "Hello, World!",
                    "name": "output_1",
                    "description": "This is a test output.",
                    "format": "{{ output_1 }}!!",
                },
                {
                    "value": "42",
                    "name": "output_2",
                    "description": "This is another test output.",
                    "format": "{{ output_2 }}",
                },
                {
                    "value": MockObject(value="!!", key="key"),
                    "name": "output_3",
                    "description": "This is a test output with a mock object.",
                    "format": "{{ output_3 }}",
                },
            ],
            test_output=[
                ("output", "Hello, World!!!"),
                ("output", "42"),
                ("output", MockObject(value="!!", key="key")),
            ],
            categories={BlockCategory.OUTPUT, BlockCategory.BASIC},
            block_type=BlockType.OUTPUT,
            static_output=True,
        )

    async def run(self, input_data: Input, *args, **kwargs) -> BlockOutput:
        """
        Attempts to format the recorded_value using the fmt_string if provided.
        If formatting fails or no fmt_string is given, returns the original recorded_value.
        """
        if input_data.format:
            try:
                formatter = TextFormatter(autoescape=input_data.escape_html)
                yield "output", formatter.format_string(
                    input_data.format, {input_data.name: input_data.value}
                )
            except Exception as e:
                yield "output", f"Error: {e}, {input_data.value}"
        else:
            yield "output", input_data.value
            yield "name", input_data.name


class AgentShortTextInputBlock(AgentInputBlock):
    class Input(AgentInputBlock.Input):
        value: Optional[ShortTextType] = SchemaField(
            description="Short text input.",
            default=None,
            advanced=False,
            title="Default Value",
        )

    class Output(AgentInputBlock.Output):
        result: str = SchemaField(description="Short text result.")

    def __init__(self):
        super().__init__(
            id="7fcd3bcb-8e1b-4e69-903d-32d3d4a92158",
            description="Block for short text input (single-line).",
            disabled=not config.enable_agent_input_subtype_blocks,
            input_schema=AgentShortTextInputBlock.Input,
            output_schema=AgentShortTextInputBlock.Output,
            test_input=[
                {
                    "value": "Hello",
                    "name": "short_text_1",
                    "description": "Short text example 1",
                    "placeholder_values": [],
                },
                {
                    "value": "Quick test",
                    "name": "short_text_2",
                    "description": "Short text example 2",
                    "placeholder_values": ["Quick test", "Another option"],
                },
            ],
            test_output=[
                ("result", "Hello"),
                ("result", "Quick test"),
            ],
        )


class AgentLongTextInputBlock(AgentInputBlock):
    class Input(AgentInputBlock.Input):
        value: Optional[LongTextType] = SchemaField(
            description="Long text input (potentially multi-line).",
            default=None,
            advanced=False,
            title="Default Value",
        )

    class Output(AgentInputBlock.Output):
        result: str = SchemaField(description="Long text result.")

    def __init__(self):
        super().__init__(
            id="90a56ffb-7024-4b2b-ab50-e26c5e5ab8ba",
            description="Block for long text input (multi-line).",
            disabled=not config.enable_agent_input_subtype_blocks,
            input_schema=AgentLongTextInputBlock.Input,
            output_schema=AgentLongTextInputBlock.Output,
            test_input=[
                {
                    "value": "Lorem ipsum dolor sit amet...",
                    "name": "long_text_1",
                    "description": "Long text example 1",
                    "placeholder_values": [],
                },
                {
                    "value": "Another multiline text input.",
                    "name": "long_text_2",
                    "description": "Long text example 2",
                    "placeholder_values": ["Another multiline text input."],
                },
            ],
            test_output=[
                ("result", "Lorem ipsum dolor sit amet..."),
                ("result", "Another multiline text input."),
            ],
        )


class AgentNumberInputBlock(AgentInputBlock):
    class Input(AgentInputBlock.Input):
        value: Optional[int] = SchemaField(
            description="Number input.",
            default=None,
            advanced=False,
            title="Default Value",
        )

    class Output(AgentInputBlock.Output):
        result: int = SchemaField(description="Number result.")

    def __init__(self):
        super().__init__(
            id="96dae2bb-97a2-41c2-bd2f-13a3b5a8ea98",
            description="Block for number input.",
            disabled=not config.enable_agent_input_subtype_blocks,
            input_schema=AgentNumberInputBlock.Input,
            output_schema=AgentNumberInputBlock.Output,
            test_input=[
                {
                    "value": 42,
                    "name": "number_input_1",
                    "description": "Number example 1",
                    "placeholder_values": [],
                },
                {
                    "value": 314,
                    "name": "number_input_2",
                    "description": "Number example 2",
                    "placeholder_values": [314, 2718],
                },
            ],
            test_output=[
                ("result", 42),
                ("result", 314),
            ],
        )


class AgentDateInputBlock(AgentInputBlock):
    class Input(AgentInputBlock.Input):
        value: Optional[date] = SchemaField(
            description="Date input (YYYY-MM-DD).",
            default=None,
            advanced=False,
            title="Default Value",
        )

    class Output(AgentInputBlock.Output):
        result: date = SchemaField(description="Date result.")

    def __init__(self):
        super().__init__(
            id="7e198b09-4994-47db-8b4d-952d98241817",
            description="Block for date input.",
            disabled=not config.enable_agent_input_subtype_blocks,
            input_schema=AgentDateInputBlock.Input,
            output_schema=AgentDateInputBlock.Output,
            test_input=[
                {
                    # If your system can parse JSON date strings to date objects
                    "value": str(date(2025, 3, 19)),
                    "name": "date_input_1",
                    "description": "Example date input 1",
                },
                {
                    "value": str(date(2023, 12, 31)),
                    "name": "date_input_2",
                    "description": "Example date input 2",
                },
            ],
            test_output=[
                ("result", date(2025, 3, 19)),
                ("result", date(2023, 12, 31)),
            ],
        )


class AgentTimeInputBlock(AgentInputBlock):
    class Input(AgentInputBlock.Input):
        value: Optional[time] = SchemaField(
            description="Time input (HH:MM:SS).",
            default=None,
            advanced=False,
            title="Default Value",
        )

    class Output(AgentInputBlock.Output):
        result: time = SchemaField(description="Time result.")

    def __init__(self):
        super().__init__(
            id="2a1c757e-86cf-4c7e-aacf-060dc382e434",
            description="Block for time input.",
            disabled=not config.enable_agent_input_subtype_blocks,
            input_schema=AgentTimeInputBlock.Input,
            output_schema=AgentTimeInputBlock.Output,
            test_input=[
                {
                    "value": str(time(9, 30, 0)),
                    "name": "time_input_1",
                    "description": "Time example 1",
                },
                {
                    "value": str(time(23, 59, 59)),
                    "name": "time_input_2",
                    "description": "Time example 2",
                },
            ],
            test_output=[
                ("result", time(9, 30, 0)),
                ("result", time(23, 59, 59)),
            ],
        )


class AgentFileInputBlock(AgentInputBlock):
    """
    A simplified file-upload block. In real usage, you might have a custom
    file type or handle binary data. Here, we'll store a string path as the example.
    """

    class Input(AgentInputBlock.Input):
        value: Optional[MediaFileType] = SchemaField(
            description="Path or reference to an uploaded file.",
            default=None,
            advanced=False,
            title="Default Value",
        )
        base_64: bool = SchemaField(
            description="Whether produce an output in base64 format (not recommended, you can pass the string path just fine accross blocks).",
            default=False,
            advanced=True,
            title="Produce Base64 Output",
        )

    class Output(AgentInputBlock.Output):
        result: str = SchemaField(description="File reference/path result.")

    def __init__(self):
        super().__init__(
            id="95ead23f-8283-4654-aef3-10c053b74a31",
            description="Block for file upload input (string path for example).",
            disabled=not config.enable_agent_input_subtype_blocks,
            input_schema=AgentFileInputBlock.Input,
            output_schema=AgentFileInputBlock.Output,
            test_input=[
                {
                    "value": "data:image/png;base64,MQ==",
                    "name": "file_upload_1",
                    "description": "Example file upload 1",
                },
            ],
            test_output=[
                ("result", str),
            ],
        )

    async def run(
        self,
        input_data: Input,
        *,
        graph_exec_id: str,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        if not input_data.value:
            return

        yield "result", await store_media_file(
            graph_exec_id=graph_exec_id,
            file=input_data.value,
            user_id=user_id,
            return_content=input_data.base_64,
        )


class AgentDropdownInputBlock(AgentInputBlock):
    """
    A specialized text input block that relies on placeholder_values to present a dropdown.
    """

    class Input(AgentInputBlock.Input):
        value: Optional[str] = SchemaField(
            description="Text selected from a dropdown.",
            default=None,
            advanced=False,
            title="Default Value",
        )
        placeholder_values: list = SchemaField(
            description="Possible values for the dropdown.",
            default_factory=list,
            advanced=False,
            title="Dropdown Options",
        )

    class Output(AgentInputBlock.Output):
        result: str = SchemaField(description="Selected dropdown value.")

    def __init__(self):
        super().__init__(
            id="655d6fdf-a334-421c-b733-520549c07cd1",
            description="Block for dropdown text selection.",
            disabled=not config.enable_agent_input_subtype_blocks,
            input_schema=AgentDropdownInputBlock.Input,
            output_schema=AgentDropdownInputBlock.Output,
            test_input=[
                {
                    "value": "Option A",
                    "name": "dropdown_1",
                    "placeholder_values": ["Option A", "Option B", "Option C"],
                    "description": "Dropdown example 1",
                },
                {
                    "value": "Option C",
                    "name": "dropdown_2",
                    "placeholder_values": ["Option A", "Option B", "Option C"],
                    "description": "Dropdown example 2",
                },
            ],
            test_output=[
                ("result", "Option A"),
                ("result", "Option C"),
            ],
        )


class AgentToggleInputBlock(AgentInputBlock):
    class Input(AgentInputBlock.Input):
        value: bool = SchemaField(
            description="Boolean toggle input.",
            default=False,
            advanced=False,
            title="Default Value",
        )

    class Output(AgentInputBlock.Output):
        result: bool = SchemaField(description="Boolean toggle result.")

    def __init__(self):
        super().__init__(
            id="cbf36ab5-df4a-43b6-8a7f-f7ed8652116e",
            description="Block for boolean toggle input.",
            disabled=not config.enable_agent_input_subtype_blocks,
            input_schema=AgentToggleInputBlock.Input,
            output_schema=AgentToggleInputBlock.Output,
            test_input=[
                {
                    "value": True,
                    "name": "toggle_1",
                    "description": "Toggle example 1",
                },
                {
                    "value": False,
                    "name": "toggle_2",
                    "description": "Toggle example 2",
                },
            ],
            test_output=[
                ("result", True),
                ("result", False),
            ],
        )


class AgentTableInputBlock(AgentInputBlock):
    """
    This block allows users to input data in a table format.

    Configure the table columns at build time, then users can input
    rows of data at runtime. Each row is output as a dictionary
    with column names as keys.
    """

    class Input(AgentInputBlock.Input):
        value: Optional[list[dict[str, Any]]] = SchemaField(
            description="The table data as a list of dictionaries.",
            default=None,
            advanced=False,
            title="Default Value",
        )
        column_headers: list[str] = SchemaField(
            description="Column headers for the table.",
            default_factory=lambda: ["Column 1", "Column 2", "Column 3"],
            advanced=False,
            title="Column Headers",
        )

        def generate_schema(self):
            """Generate schema for the value field with table format."""
            schema = super().generate_schema()
            schema["type"] = "array"
            schema["format"] = "table"
            schema["items"] = {
                "type": "object",
                "properties": {
                    header: {"type": "string"}
                    for header in (
                        self.column_headers or ["Column 1", "Column 2", "Column 3"]
                    )
                },
            }
            if self.value is not None:
                schema["default"] = self.value
            return schema

    class Output(AgentInputBlock.Output):
        result: list[dict[str, Any]] = SchemaField(
            description="The table data as a list of dictionaries with headers as keys."
        )

    def __init__(self):
        super().__init__(
            id="5603b273-f41e-4020-af7d-fbc9c6a8d928",
            description="Block for table data input with customizable headers.",
            disabled=not config.enable_agent_input_subtype_blocks,
            input_schema=AgentTableInputBlock.Input,
            output_schema=AgentTableInputBlock.Output,
            test_input=[
                {
                    "name": "test_table",
                    "column_headers": ["Name", "Age", "City"],
                    "value": [
                        {"Name": "John", "Age": "30", "City": "New York"},
                        {"Name": "Jane", "Age": "25", "City": "London"},
                    ],
                    "description": "Example table input",
                }
            ],
            test_output=[
                (
                    "result",
                    [
                        {"Name": "John", "Age": "30", "City": "New York"},
                        {"Name": "Jane", "Age": "25", "City": "London"},
                    ],
                )
            ],
        )

    async def run(self, input_data: Input, *args, **kwargs) -> BlockOutput:
        """
        Yields the table data as a list of dictionaries.
        """
        # Pass through the value, defaulting to empty list if None
        yield "result", input_data.value if input_data.value is not None else []


IO_BLOCK_IDs = [
    AgentInputBlock().id,
    AgentOutputBlock().id,
    AgentShortTextInputBlock().id,
    AgentLongTextInputBlock().id,
    AgentNumberInputBlock().id,
    AgentDateInputBlock().id,
    AgentTimeInputBlock().id,
    AgentFileInputBlock().id,
    AgentDropdownInputBlock().id,
    AgentToggleInputBlock().id,
    AgentTableInputBlock().id,
]
