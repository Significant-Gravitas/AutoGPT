import uuid
from typing import Any, List, Optional

import chromadb
from chromadb.config import Settings
from pydantic import Field

from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.data.model import BlockSecret, SchemaField, SecretField
from autogpt_server.blocks.llm import AIStructuredResponseGeneratorBlock, LlmModel


class EpisodicMemory:
    def __init__(self):
        self.client = chromadb.Client(Settings(is_persistent=True))
        try:
            self.collection = self.client.get_collection("episodic_memory")
        except ValueError:
            self.collection = self.client.create_collection("episodic_memory")

    def add_memory(self, content: str, metadata: Optional[dict] = None) -> str:
        memory_id = str(uuid.uuid4())
        self.collection.add(
            documents=[content],
            metadatas=[metadata or {}],
            ids=[memory_id]
        )
        return memory_id

    def query_memory(self, query: str, n_results: int = 5) -> List[dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return [
            {"id": id, "content": doc, "metadata": meta}
            for id, doc, meta in zip(results['ids'][0], results['documents'][0], results['metadatas'][0])
        ]


class MemoryAugmentedAIBlock(Block):
    class Input(BlockSchema):
        prompt: str = SchemaField(description="The main prompt for the AI")
        memory_query: str = SchemaField(description="Query to retrieve relevant memories")
        model: LlmModel = LlmModel.GPT4_TURBO
        api_key: BlockSecret = SecretField(value="")
        sys_prompt: str = ""
        expected_format: dict[str, str] = Field(default_factory=dict)
        retry: int = 3
        n_memories: int = SchemaField(default=3, description="Number of memories to retrieve")

    class Output(BlockSchema):
        response: dict[str, str] = SchemaField(description="The AI's response")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="2c3c15d1-387a-4124-a087-39999f2f3b09",
            description="This block augments AI responses with episodic memory.",
            categories={BlockCategory.AI, BlockCategory.MEMORY},
            input_schema=MemoryAugmentedAIBlock.Input,
            output_schema=MemoryAugmentedAIBlock.Output
        )
        self.memory = EpisodicMemory()
        self.ai_block = AIStructuredResponseGeneratorBlock()

    def run(self, input_data: Input) -> BlockOutput:
        try:
            # Retrieve relevant memories
            memories = self.memory.query_memory(input_data.memory_query, input_data.n_memories)

            # Format memories for inclusion in the prompt
            formatted_memories = "\n".join([f"Memory {i + 1}: {m['content']}" for i, m in enumerate(memories)])

            # Augment the prompt with memories
            augmented_prompt = f"""Relevant memories:
{formatted_memories}

Using the above memories if they are relevant, please respond to the following prompt:
{input_data.prompt}"""

            # Prepare input for the AI block
            ai_input = AIStructuredResponseGeneratorBlock.Input(
                prompt=augmented_prompt,
                expected_format=input_data.expected_format,
                model=input_data.model,
                api_key=input_data.api_key,
                sys_prompt=input_data.sys_prompt,
                retry=input_data.retry
            )

            # Run the AI block
            for output_name, output_data in self.ai_block.run(ai_input):
                if output_name == "response":
                    # Add the response to memory
                    self.memory.add_memory(str(output_data), {"prompt": input_data.prompt})
                    yield "response", output_data
                elif output_name == "error":
                    yield "error", output_data

        except Exception as e:
            yield "error", str(e)