import dataclasses

import numpy as np

from autogpt.config import Config
from autogpt.logs import logger
from autogpt.processing.text import chunk_text, summarize_text

from .utils import Embedding, get_embedding


@dataclasses.dataclass
class MemoryItem:
    """Memory object containing raw content as well as embeddings"""

    raw_content: str
    e_summary: Embedding
    e_chunks: list[Embedding]
    metadata: dict

    def from_text(text: str, metadata: dict = {}):
        cfg = Config()
        logger.debug(f"Memorizing text:\n{'-'*32}\n{text}\n{'-'*32}\n")

        chunks = chunk_text(text, cfg.embedding_model)
        logger.debug("Chunks: " + str(chunks))

        chunk_summaries = [summarize_text(text_chunk) for text_chunk in chunks]
        logger.debug("Chunk summaries: " + str(chunk_summaries))

        e_chunks = get_embedding(chunks)
        logger.debug("Chunk embeddings: " + str(e_chunks))

        summary = (
            chunk_summaries[0]
            if len(chunks) == 1
            else summarize_text("\n\n".join(chunk_summaries))
        )
        logger.debug("Total summary: " + summary)

        # TODO: investigate search performance of weighted average vs summary
        # e_average = np.average(e_chunks, axis=0, weights=[len(c) for c in chunks])
        e_summary = get_embedding(summary)

        return MemoryItem(
            text,
            e_summary,
            e_chunks,
            metadata=metadata,
        )
