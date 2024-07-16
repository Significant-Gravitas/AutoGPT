import math
from typing import List
from autogpt_server.data.block import Block, BlockSchema, BlockOutput, BlockFieldSecret
from autogpt_server.blocks.llm import LlmModel, LlmCallBlock

class Summariser(Block):
    class Input(BlockSchema):
        text: str
        api_key: BlockFieldSecret = BlockFieldSecret(key="openai_api_key")
        model: LlmModel = LlmModel.openai_gpt4
        max_tokens: int = 4000  # Adjust based on the model's context window TODO: Make this dynamic and hide from the user
        chunk_overlap: int = 100  # Overlap between chunks to maintain context

    class Output(BlockSchema):
        summary: str

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-7g8h-9i0j-1k2l-m3n4o5p6q7r8",
            input_schema=Summariser.Input,
            output_schema=Summariser.Output,
            test_input={
                "text": "Lorem ipsum..." * 1000,  # Long text for testing
                "api_key": "fake-api-key",
                "model": LlmModel.openai_gpt4,
            },
            test_output={"summary": "This is a summary of the long text..."},
        )

    def run(self, input_data: Input) -> BlockOutput:
        llm_block = LlmCallBlock()
        chunks = self._split_text(input_data.text, input_data.max_tokens, input_data.chunk_overlap)
        summaries = []

        for chunk in chunks:
            chunk_summary = self._summarize_chunk(chunk, llm_block, input_data)
            summaries.append(chunk_summary)

        final_summary = self._combine_summaries(summaries, llm_block, input_data)
        yield "summary", final_summary

    def _split_text(self, text: str, max_tokens: int, overlap: int) -> List[str]:
        words = text.split()
        chunks = []
        chunk_size = max_tokens - overlap
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + max_tokens])
            chunks.append(chunk)
        
        return chunks

    def _summarize_chunk(self, chunk: str, llm_block: LlmCallBlock, input_data: Input) -> str:
        prompt = f"Summarize the following text concisely:\n\n{chunk}"
        
        llm_input = llm_block.Input(
            prompt=prompt,
            api_key=input_data.api_key,
            model=input_data.model,
            expected_format={"summary": "The summary of the given text."}
        )
        
        for output_name, output_data in llm_block.run(llm_input):
            if output_name == "response":
                return output_data["summary"]
        
        raise ValueError("Failed to get a summary from the LLM.")

    def _combine_summaries(self, summaries: List[str], llm_block: LlmCallBlock, input_data: Input) -> str:
        combined_text = " ".join(summaries)
        
        if len(combined_text.split()) <= input_data.max_tokens:
            prompt = f"Provide a final, concise summary of the following summaries:\n\n{combined_text}"
            
            llm_input = llm_block.Input(
                prompt=prompt,
                api_key=input_data.api_key,
                model=input_data.model,
                expected_format={"final_summary": "The final summary of all provided summaries."}
            )
            
            for output_name, output_data in llm_block.run(llm_input):
                if output_name == "response":
                    return output_data["final_summary"]
            
            raise ValueError("Failed to get a final summary from the LLM.")
        else:
            # If combined summaries are still too long, recursively summarize
            return self.run(Summariser.Input(
                text=combined_text,
                api_key=input_data.api_key,
                model=input_data.model,
                max_tokens=input_data.max_tokens,
                chunk_overlap=input_data.chunk_overlap
            )).send(None)[1]  # Get the first yielded value