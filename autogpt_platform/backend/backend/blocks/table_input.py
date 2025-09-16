from typing import Any, Dict, List, Optional

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema, BlockType
from backend.data.model import SchemaField


class TableInputBlock(Block):
    """
    This block allows users to input data in a table format.
    
    The block takes headers (column names) as input and provides a table UI 
    where users can add rows of data. Each row is output as a dictionary 
    with headers as keys.
    """

    class Input(BlockSchema):
        name: str = SchemaField(
            description="The name of the table input.",
            default="table_input"
        )
        headers: List[str] = SchemaField(
            description="List of column headers for the table.",
            default_factory=lambda: ["Column 1", "Column 2", "Column 3"]
        )
        value: List[Dict[str, Any]] = SchemaField(
            description="The table data as a list of dictionaries.",
            default_factory=list,
            advanced=False
        )
        title: Optional[str] = SchemaField(
            description="The title of the table input.",
            default=None,
            advanced=True
        )
        description: Optional[str] = SchemaField(
            description="The description of the table input.",
            default=None,
            advanced=True
        )
        advanced: bool = SchemaField(
            description="Whether to show the input in the advanced section.",
            default=False,
            advanced=True
        )
        
        # Flag to indicate this is a table input
        is_table_input: bool = SchemaField(
            description="Internal flag to indicate table input UI should be used.",
            default=True,
            hidden=True
        )

    class Output(BlockSchema):
        result: List[Dict[str, Any]] = SchemaField(
            description="The table data as a list of dictionaries with headers as keys."
        )

    def __init__(self, **kwargs):
        super().__init__(
            **{
                "id": "5603b273-f41e-4020-af7d-fbc9c6a8d928",
                "description": "Input data in a table format with customizable headers.",
                "input_schema": TableInputBlock.Input,
                "output_schema": TableInputBlock.Output,
                "test_input": [
                    {
                        "name": "test_table",
                        "headers": ["Name", "Age", "City"],
                        "value": [
                            {"Name": "John", "Age": "30", "City": "New York"},
                            {"Name": "Jane", "Age": "25", "City": "London"}
                        ]
                    }
                ],
                "test_output": [
                    ("result", [
                        {"Name": "John", "Age": "30", "City": "New York"},
                        {"Name": "Jane", "Age": "25", "City": "London"}
                    ])
                ],
                "categories": {BlockCategory.INPUT, BlockCategory.BASIC},
                "block_type": BlockType.INPUT,
                "static_output": True,
                **kwargs,
            }
        )

    async def run(self, input_data: Input, *args, **kwargs) -> BlockOutput:
        """
        Yields the table data as a list of dictionaries.
        
        Each dictionary represents a row with headers as keys and 
        corresponding cell values as values.
        """
        # Validate headers
        if not input_data.headers:
            yield "error", "Headers cannot be empty"
            return
        
        # Check for duplicate headers
        if len(input_data.headers) != len(set(input_data.headers)):
            yield "error", "Duplicate headers are not allowed"
            return
        
        if input_data.value:
            # Ensure all rows have all headers with at least empty strings
            normalized_data = []
            for row in input_data.value:
                if not isinstance(row, dict):
                    continue
                normalized_row = {}
                for header in input_data.headers:
                    normalized_row[header] = row.get(header, "")
                normalized_data.append(normalized_row)
            
            yield "result", normalized_data
        else:
            # Return empty list if no data
            yield "result", []