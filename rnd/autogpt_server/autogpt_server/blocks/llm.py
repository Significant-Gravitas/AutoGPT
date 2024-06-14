from typing import ClassVar, Any
from autogpt_server.data.block import Block, BlockSchema, BlockData
from openai import OpenAI


class LlmBlock(Block):
    id: ClassVar[str] = "db7d8f02-2f44-4c55-ab7a-eae0941f0c33"  # type: ignore
    input_schema: ClassVar[BlockSchema] = BlockSchema(
        {  # type: ignore
            "model": "string",
            "messages": "Dict[string, Any]",
        }
    )
    output_schema: ClassVar[BlockSchema] = BlockSchema(
        {  # type: ignore
            "response": "string",
        }
    )

    async def run(self, input_data: BlockData) -> tuple[str, Any]:
        client = OpenAI()

        completion = client.chat.completions.create(
            model=input_data["model"], messages=input_data["messages"]
        )
        return "response", completion.choices[0].message
