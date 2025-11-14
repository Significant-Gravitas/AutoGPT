from itertools import chain
from backend.data.block import BlockIOBase, BlockIOType, SchemaField
from backend.data.block_decorator import block


@block(
    name="Concatenate Lists",
    description="Concatenates two or more lists into one",
    category="Lists",
    input_schema={
        "lists": SchemaField(
            type=BlockIOType.LIST,
            description="List of lists to concatenate",
        ),
    },
    output_schema={
        "result": SchemaField(
            type=BlockIOType.LIST,
            description="Concatenated list result",
        ),
    },
)
class ConcatenateListsBlock(BlockIOBase):
    def run(self, lists: list) -> dict:
        """Merge multiple lists into a single list."""
        try:
            if not isinstance(lists, list):
                raise ValueError("Input must be a list of lists")

            for lst in lists:
                if not isinstance(lst, list):
                    raise ValueError("All elements inside 'lists' must be lists")

            result = list(chain.from_iterable(lists))
            return {"result": result}

        except Exception as e:
            return {"error": str(e)}
