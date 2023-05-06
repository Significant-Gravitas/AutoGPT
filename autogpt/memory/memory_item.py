import dataclasses
from typing import Literal

import numpy as np

from autogpt.config import Config
from autogpt.logs import logger
from autogpt.processing.text import chunk_content, split_text, summarize_text

from .utils import Embedding, get_embedding

MemoryDocType = Literal["webpage", "text_file", "code_file"]


@dataclasses.dataclass
class MemoryItem:
    """Memory object containing raw content as well as embeddings"""

    raw_content: str
    summary: str
    e_summary: Embedding
    e_chunks: list[Embedding]
    metadata: dict

    @staticmethod
    def from_text(text: str, source_type: MemoryDocType, metadata: dict = {}):
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
            for summary, _ in [summarize_text(text_chunk) for text_chunk in chunks]
        ]
        logger.debug("Chunk summaries: " + str(chunk_summaries))

        e_chunks = get_embedding(chunks)

        summary = (
            chunk_summaries[0]
            if len(chunks) == 1
            else summarize_text("\n\n".join(chunk_summaries))[0]
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
    def from_webpage(content: str, url: str):
        return MemoryItem.from_text(content, "webpage", {"location": url})
