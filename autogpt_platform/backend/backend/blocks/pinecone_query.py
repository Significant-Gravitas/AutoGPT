from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class PineconeQueryBlock(Block):
    class Input(BlockSchema):
        index: object = SchemaField(description="Initialized Pinecone index")
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
        results = input_data.index.query(
            namespace=input_data.namespace,
            vector=input_data.query_vector,
            top_k=input_data.top_k,
            include_values=input_data.include_values,
            include_metadata=input_data.include_metadata,
        )
        yield "results", results
