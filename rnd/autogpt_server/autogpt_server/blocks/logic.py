from typing import ClassVar, Any
from autogpt_server.data.block import Block, BlockSchema, BlockData


class IfBlock(Block):
    id: ClassVar[str] = "db7d8f02-2f44-4c55-ab7a-eae0941f0c31"  # type: ignore
    input_schema: ClassVar[BlockSchema] = BlockSchema(
        {  # type: ignore
            "condition": "string",
            "variables": "Dict[string, Any]",
        }
    )
    output_schema: ClassVar[BlockSchema] = BlockSchema(
        {  # type: ignore
            "result": "bool",
        }
    )

    async def run(self, input_data: BlockData) -> tuple[str, Any]:
        for key in input_data.keys():
            eval(f"{key} = {input_data[key]}")
        if eval(input_data["condition"]):
            return "result", True
        else:
            return "result", False


class ForBlock(Block):
    id: ClassVar[str] = "db7d8f02-2f44-4c55-ab7a-eae0941f0c32"  # type: ignore
    input_schema: ClassVar[BlockSchema] = BlockSchema(
        {  # type: ignore
            "data": "List[Any]"
        }
    )
    output_schema: ClassVar[BlockSchema] = BlockSchema(
        {  # type: ignore
            "result": "Any",
        }
    )

    async def run(self, input_data: BlockData) -> tuple[str, Any]:
        # FIX: Current block does not support multiple returns...:W
        for element in input_data["data"]:
            return "result", element
