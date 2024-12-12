import uuid
from typing import Any, Literal

from pinecone import Pinecone, ServerlessSpec

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName

PineconeCredentials = APIKeyCredentials
PineconeCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.PINECONE],
    Literal["api_key"],
]


def PineconeCredentialsField() -> PineconeCredentialsInput:
    """Creates a Pinecone credentials input on a block."""
    return CredentialsField(
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
        host: str = SchemaField(description="Host for pinecone", default="")
        idx_name: str = SchemaField(description="Index name for pinecone")

    class Output(BlockSchema):
        results: Any = SchemaField(description="Query results from Pinecone")
        combined_results: Any = SchemaField(
            description="Combined results from Pinecone"
        )

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
        try:
            # Create a new client instance
            pc = Pinecone(api_key=credentials.api_key.get_secret_value())

            # Get the index
            idx = pc.Index(input_data.idx_name)

            # Ensure query_vector is in correct format
            query_vector = input_data.query_vector
            if isinstance(query_vector, list) and len(query_vector) > 0:
                if isinstance(query_vector[0], list):
                    query_vector = query_vector[0]

            results = idx.query(
                namespace=input_data.namespace,
                vector=query_vector,
                top_k=input_data.top_k,
                include_values=input_data.include_values,
                include_metadata=input_data.include_metadata,
            ).to_dict()  # type: ignore
            combined_text = ""
            if results["matches"]:
                texts = [
                    match["metadata"]["text"]
                    for match in results["matches"]
                    if match.get("metadata", {}).get("text")
                ]
                combined_text = "\n\n".join(texts)

            # Return both the raw matches and combined text
            yield "results", {
                "matches": results["matches"],
                "combined_text": combined_text,
            }
            yield "combined_results", combined_text

        except Exception as e:
            error_msg = f"Error querying Pinecone: {str(e)}"
            raise RuntimeError(error_msg) from e


class PineconeInsertBlock(Block):
    class Input(BlockSchema):
        credentials: PineconeCredentialsInput = PineconeCredentialsField()
        index: str = SchemaField(description="Initialized Pinecone index")
        chunks: list = SchemaField(description="List of text chunks to ingest")
        embeddings: list = SchemaField(
            description="List of embeddings corresponding to the chunks"
        )
        namespace: str = SchemaField(
            description="Namespace to use in Pinecone", default=""
        )
        metadata: dict = SchemaField(
            description="Additional metadata to store with each vector", default={}
        )

    class Output(BlockSchema):
        upsert_response: str = SchemaField(
            description="Response from Pinecone upsert operation"
        )

    def __init__(self):
        super().__init__(
            id="477f2168-cd91-475a-8146-9499a5982434",
            description="Upload data to a Pinecone index",
            categories={BlockCategory.LOGIC},
            input_schema=PineconeInsertBlock.Input,
            output_schema=PineconeInsertBlock.Output,
        )

    def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            # Create a new client instance
            pc = Pinecone(api_key=credentials.api_key.get_secret_value())

            # Get the index
            idx = pc.Index(input_data.index)

            vectors = []
            for chunk, embedding in zip(input_data.chunks, input_data.embeddings):
                vector_metadata = input_data.metadata.copy()
                vector_metadata["text"] = chunk
                vectors.append(
                    {
                        "id": str(uuid.uuid4()),
                        "values": embedding,
                        "metadata": vector_metadata,
                    }
                )
            idx.upsert(vectors=vectors, namespace=input_data.namespace)

            yield "upsert_response", "successfully upserted"

        except Exception as e:
            error_msg = f"Error uploading to Pinecone: {str(e)}"
            raise RuntimeError(error_msg) from e
