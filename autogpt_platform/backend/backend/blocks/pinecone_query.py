from pinecone import Pinecone

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField


class PineconeQueryBlock(Block):
    class Input(BlockSchema):
        api_key: BlockSecret = SecretField(
            key="pinecone_api_key", description="Pinecone API Key"
        )
        query_vector: list = SchemaField(description="Query vector")
        namespace: str = SchemaField(
            description="Namespace to query in Pinecone", default=""
        )
        top_k: int = SchemaField(
            description="Number of top results to return", default=3
        )
        include_values: bool = SchemaField(
            description="Whether to include vector values in the response",
            default=False,
        )
        include_metadata: bool = SchemaField(
            description="Whether to include metadata in the response", default=True
        )
        host: str = SchemaField(description="Host for pinecone")

    class Output(BlockSchema):
        results: dict = SchemaField(description="Query results from Pinecone")

    def __init__(self):
        super().__init__(
            id="9ad93d0f-91b4-4c9c-8eb1-82e26b4a01c5",
            description="Queries a Pinecone index",
            categories={BlockCategory.LOGIC},
            input_schema=PineconeQueryBlock.Input,
            output_schema=PineconeQueryBlock.Output,
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        pc = Pinecone(api_key=input_data.api_key.get_secret_value())
        idx = pc.Index(host=input_data.host)
        results = idx.query(
            namespace=input_data.namespace,
            vector=input_data.query_vector,
            top_k=input_data.top_k,
            include_values=input_data.include_values,
            include_metadata=input_data.include_metadata,
        )
        yield "results", results
