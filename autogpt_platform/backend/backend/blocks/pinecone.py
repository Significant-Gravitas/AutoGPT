from typing import Literal

from autogpt_libs.supabase_integration_credentials_store import APIKeyCredentials
from pinecone import Pinecone, ServerlessSpec

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import CredentialsField, CredentialsMetaInput, SchemaField

PineconeCredentials = APIKeyCredentials
PineconeCredentialsInput = CredentialsMetaInput[
    Literal["pinecone"],
    Literal["api_key"],
]


def PineconeCredentialsField() -> PineconeCredentialsInput:
    """
    Creates a Pinecone credentials input on a block.

    """
    return CredentialsField(
        provider="pinecone",
        supported_credential_types={"api_key"},
        description="The Pinecone integration can be used with an API Key.",
    )


class PineconeInitBlock(Block):
    class Input(BlockSchema):
        credentials: PineconeCredentialsInput = PineconeCredentialsField()
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

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        pc = Pinecone(api_key=credentials.api_key.get_secret_value())

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


class PineconeQueryBlock(Block):
    class Input(BlockSchema):
        credentials: PineconeCredentialsInput = PineconeCredentialsField()
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

    def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        pc = Pinecone(api_key=credentials.api_key.get_secret_value())
        idx = pc.Index(host=input_data.host)
        results = idx.query(
            namespace=input_data.namespace,
            vector=input_data.query_vector,
            top_k=input_data.top_k,
            include_values=input_data.include_values,
            include_metadata=input_data.include_metadata,
        )
        yield "results", results
