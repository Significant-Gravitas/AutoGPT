from autogpt_server.data.block import Block, BlockSchema, BlockOutput

from typing import Any


class ConstantBlock(Block):
    """
    This block allows you to provide a constant value as a block, in a stateless manner.
    The common use-case is simply pass the `input` data, it will `output` the same data.
    But this will not retain the state, once it is executed, the output is consumed. 

    To retain the state, you can feed the `output` to the `data` input, so that the data
    is retained in the block for the next execution. You can then trigger the block by
    feeding the `input` pin with any data, and the block will produce value of `data`.
    
    Ex:
         <constant_data>  <any_trigger>
                ||           ||   
       =====> `data`      `input`
      ||        \\         //
      ||       ConstantBlock
      ||           ||
       ======  `output`
    """
    class Input(BlockSchema):
        input: Any
        data: Any = None

    class Output(BlockSchema):
        output: Any

    def __init__(self):
        super().__init__(
            id="1ff065e9-88e8-4358-9d82-8dc91f622ba9",
            input_schema=ConstantBlock.Input,
            output_schema=ConstantBlock.Output,
            test_input=[
                {"input": "Hello, World!"},
                {"input": "Hello, World!", "data": "Existing Data"},
            ],
            test_output=[
                ("output", "Hello, World!"),  # No data provided, so trigger is returned
                ("output", "Existing Data"),  # Data is provided, so data is returned.
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        yield "output", input_data.data or input_data.input


class PrintingBlock(Block):
    class Input(BlockSchema):
        text: str

    class Output(BlockSchema):
        status: str

    def __init__(self):
        super().__init__(
            id="f3b1c1b2-4c4f-4f0d-8d2f-4c4f0d8d2f4c",
            input_schema=PrintingBlock.Input,
            output_schema=PrintingBlock.Output,
            test_input={"text": "Hello, World!"},
            test_output=("status", "printed"),
        )

    def run(self, input_data: Input) -> BlockOutput:
        print(">>>>> Print: ", input_data.text)
        yield "status", "printed"
