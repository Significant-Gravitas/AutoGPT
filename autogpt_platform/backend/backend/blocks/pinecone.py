from pinecone import Pinecone, ServerlessSpec

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField


class PineconeInitBlock(Block):
    class Input(BlockSchema):
        api_key: BlockSecret = SecretField(
            key="pinecone_api_key", description="Pinecone API Key"
        )
        index_name: str = SchemaField(description="Name of the Pinecone index")
        dimension: int = SchemaField(
            description="Dimension of the vectors", default=768
        )
        metric: str = SchemaField(
            description="Distance metric for the index", default="cosine"
        )
        cloud: str = SchemaField(
            description="Cloud provider for serverless", default="aws"
        )
        region: str = SchemaField(
            description="Region for serverless", default="us-east-1"
        )

    class Output(BlockSchema):
        index: str = SchemaField(description="Name of the initialized Pinecone index")
        message: str = SchemaField(description="Status message")

    def __init__(self):
        super().__init__(
            id="48d8fdab-8f03-41f3-8407-8107ba11ec9b",
            description="Initializes a Pinecone index",
            categories={BlockCategory.LOGIC},
            input_schema=PineconeInitBlock.Input,
            output_schema=PineconeInitBlock.Output,
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        pc = Pinecone(api_key=input_data.api_key.get_secret_value())

        try:
            existing_indexes = pc.list_indexes()
            if input_data.index_name not in [index.name for index in existing_indexes]:
                pc.create_index(
                    name=input_data.index_name,
                    dimension=input_data.dimension,
                    metric=input_data.metric,
                    spec=ServerlessSpec(
                        cloud=input_data.cloud, region=input_data.region
                    ),
                )
                message = f"Created new index: {input_data.index_name}"
            else:
                message = f"Using existing index: {input_data.index_name}"

            yield "index", input_data.index_name
            yield "message", message
        except Exception as e:
            yield "message", f"Error initializing Pinecone index: {str(e)}"
