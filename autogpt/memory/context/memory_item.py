import dataclasses
import json
from typing import Literal

import numpy as np

from autogpt.config import Config
from autogpt.llm.base import Message
from autogpt.llm.token_counter import count_string_tokens
from autogpt.logs import logger
from autogpt.processing.text import chunk_content, split_text, summarize_text

from .utils import Embedding, get_embedding

MemoryDocType = Literal["webpage", "text_file", "code_file", "agent_history"]


@dataclasses.dataclass
class MemoryItem:
    """Memory object containing raw content as well as embeddings"""

    raw_content: str
    summary: str
    e_summary: Embedding
    e_chunks: list[Embedding]
    metadata: dict

    @staticmethod
    def from_text(
        text: str,
        source_type: MemoryDocType,
        metadata: dict = {},
        summarization_instruction: str | None = None,
    ):
        cfg = Config()
        logger.debug(f"Memorizing text:\n{'-'*32}\n{text}\n{'-'*32}\n")

        chunks = [
            chunk
            for chunk, _ in (
                split_text(text, cfg.embedding_model)
                if source_type != "code_file"
                else chunk_content(text, cfg.embedding_model)
            )
        ]
        logger.debug("Chunks: " + str(chunks))

        chunk_summaries = [
            summary
            for summary, _ in [
                summarize_text(text_chunk, summarization_instruction)
                for text_chunk in chunks
            ]
        ]
        logger.debug("Chunk summaries: " + str(chunk_summaries))

        e_chunks = get_embedding(chunks)

        summary = (
            chunk_summaries[0]
            if len(chunks) == 1
            else summarize_text(
                "\n\n".join(chunk_summaries), summarization_instruction
            )[0]
        )
        logger.debug("Total summary: " + summary)

        # TODO: investigate search performance of weighted average vs summary
        # e_average = np.average(e_chunks, axis=0, weights=[len(c) for c in chunks])
        e_summary = get_embedding(summary)

        metadata["source_type"] = source_type

        return MemoryItem(
            text,
            summary,
            e_summary,
            e_chunks,
            metadata=metadata,
        )

    @staticmethod
    def from_text_file(content: str, path: str):
        return MemoryItem.from_text(content, "text_file", {"location": path})

    @staticmethod
    def from_code_file(content: str, path: str):
        # TODO: implement tailored code memories
        return MemoryItem.from_text(content, "code_file", {"location": path})

    @staticmethod
    def from_ai_action(ai_message: Message, result_message: Message):
        # The result_message contains either user feedback
        # or the result of the command specified in ai_message

        if ai_message["role"] != "assistant":
            raise ValueError(f"Invalid role on 'ai_message': {ai_message['role']}")

        result = (
            result_message["content"]
            if result_message["content"].startswith("Command")
            else "None"
        )
        user_input = (
            result_message["content"]
            if result_message["content"].startswith("Human feedback")
            else "None"
        )
        memory_content = (
            f"Assistant Reply: {ai_message['content']}"
            "\n\n"
            f"Result: {result}"
            "\n\n"
            f"Human Feedback: {user_input}"
        )

        return MemoryItem.from_text(
            memory_content,
            "agent_history",
            summarization_instruction="if possible, also make clear the link between the command in the assistant's response and the command result. Do not mention the human feedback if there is none",
        )

    @staticmethod
    def from_webpage(content: str, url: str):
        return MemoryItem.from_text(content, "webpage", {"location": url})

    def __str__(self) -> str:
        token_length = count_string_tokens(self.raw_content, Config().embedding_model)
        return f"""
=============== MemoryItem ===============
Length: {token_length} tokens in {len(self.e_chunks)} chunks
Metadata: {json.dumps(self.metadata, indent=2)}
---------------- SUMMARY -----------------
{self.summary}
------------------ RAW -------------------
{self.raw_content}
==========================================
"""
