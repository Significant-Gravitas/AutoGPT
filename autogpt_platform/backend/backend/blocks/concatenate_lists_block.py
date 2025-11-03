from backend.data.block import BlockIOBase, BlockIOType, SchemaField
from backend.data.block_decorator import block


@block(
    name="Concatenate Lists",
    description="Concatenates two or more lists into one",
    category="Lists",
    input_schema={
        "lists": SchemaField(
            type=BlockIOType.LIST,
            description="List of lists to concatenate"
        ),
    },
    output_schema={
        "result": SchemaField(
            type=BlockIOType.LIST,
            description="Concatenated list result"
        ),
    },
)
class ConcatenateListsBlock(BlockIOBase):
    def run(self, lists):
        try:
            # Combine all lists into one
            result = []
            for lst in lists:
                if isinstance(lst, list):
                    result.extend(lst)
                else:
                    raise ValueError("All inputs must be lists")
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
