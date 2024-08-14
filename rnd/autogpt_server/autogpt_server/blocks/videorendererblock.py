from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.data.model import SchemaField

class VideoRendererBlock(Block):
    class Input(BlockSchema):
        video_url: str = SchemaField(description="The URL of the video to be rendered.")

    class Output(BlockSchema):
        video_url: str = SchemaField(description="The URL of the video to be rendered.")

    def __init__(self):
        super().__init__(
            id="a92a0017-2390-425f-b5a8-fb3c50c81400",
            description="Renders a video from a given URL within the block.",
            input_schema=VideoRendererBlock.Input,
            output_schema=VideoRendererBlock.Output
        )

    def run(self, input_data: Input) -> BlockOutput:
        yield "video_url", input_data.video_url