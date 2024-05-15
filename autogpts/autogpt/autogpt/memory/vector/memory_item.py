from __future__ import annotations

import json
import logging
from typing import Literal

import ftfy
import numpy as np
from forge.config.config import Config
from forge.content_processing.text import chunk_content, split_text, summarize_text
from forge.llm.providers import ChatMessage, ChatModelProvider, EmbeddingModelProvider
from pydantic import BaseModel

from .utils import Embedding, get_embedding

logger = logging.getLogger(__name__)

MemoryDocType = Literal["webpage", "text_file", "code_file", "agent_history"]


class MemoryItem(BaseModel, arbitrary_types_allowed=True):
    """Memory object containing raw content as well as embeddings"""

    raw_content: str
    summary: str
    chunks: list[str]
    chunk_summaries: list[str]
    e_summary: Embedding
    e_chunks: list[Embedding]
    metadata: dict

    def relevance_for(self, query: str, e_query: Embedding | None = None):
        return MemoryItemRelevance.of(self, query, e_query)

    def dump(self, calculate_length=False) -> str:
        n_chunks = len(self.e_chunks)
        return f"""
=============== MemoryItem ===============
Size: {n_chunks} chunks
Metadata: {json.dumps(self.metadata, indent=2)}
---------------- SUMMARY -----------------
{self.summary}
------------------ RAW -------------------
{self.raw_content}
==========================================
"""

    def __eq__(self, other: MemoryItem):
        return (
            self.raw_content == other.raw_content
            and self.chunks == other.chunks
            and self.chunk_summaries == other.chunk_summaries
            # Embeddings can either be list[float] or np.ndarray[float32],
            # and for comparison they must be of the same type
            and np.array_equal(
                self.e_summary
                if isinstance(self.e_summary, np.ndarray)
                else np.array(self.e_summary, dtype=np.float32),
                other.e_summary
                if isinstance(other.e_summary, np.ndarray)
                else np.array(other.e_summary, dtype=np.float32),
            )
            and np.array_equal(
                self.e_chunks
                if isinstance(self.e_chunks[0], np.ndarray)
                else [np.array(c, dtype=np.float32) for c in self.e_chunks],
                other.e_chunks
                if isinstance(other.e_chunks[0], np.ndarray)
                else [np.array(c, dtype=np.float32) for c in other.e_chunks],
            )
        )


class MemoryItemFactory:
    def __init__(
        self,
        llm_provider: ChatModelProvider,
        embedding_provider: EmbeddingModelProvider,
    ):
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider

    async def from_text(
        self,
        text: str,
        source_type: MemoryDocType,
        config: Config,
        metadata: dict = {},
        how_to_summarize: str | None = None,
        question_for_summary: str | None = None,
    ):
        logger.debug(f"Memorizing text:\n{'-'*32}\n{text}\n{'-'*32}\n")

        # Fix encoding, e.g. removing unicode surrogates (see issue #778)
        text = ftfy.fix_text(text)

        # FIXME: needs ModelProvider
        chunks = [
            chunk
            for chunk, _ in (
                split_text(
                    text=text,
                    config=config,
                    max_chunk_length=1000,  # arbitrary, but shorter ~= better
                    tokenizer=self.llm_provider.get_tokenizer(config.fast_llm),
                )
                if source_type != "code_file"
                # TODO: chunk code based on structure/outline
                else chunk_content(
                    content=text,
                    max_chunk_length=1000,
                    tokenizer=self.llm_provider.get_tokenizer(config.fast_llm),
                )
            )
        ]
        logger.debug("Chunks: " + str(chunks))

        chunk_summaries = [
            summary
            for summary, _ in [
                await summarize_text(
                    text=text_chunk,
                    instruction=how_to_summarize,
                    question=question_for_summary,
                    llm_provider=self.llm_provider,
                    config=config,
                )
                for text_chunk in chunks
            ]
        ]
        logger.debug("Chunk summaries: " + str(chunk_summaries))

        e_chunks = get_embedding(chunks, config, self.embedding_provider)

        summary = (
            chunk_summaries[0]
            if len(chunks) == 1
            else (
                await summarize_text(
                    text="\n\n".join(chunk_summaries),
                    instruction=how_to_summarize,
                    question=question_for_summary,
                    llm_provider=self.llm_provider,
                    config=config,
                )
            )[0]
        )
        logger.debug("Total summary: " + summary)

        # TODO: investigate search performance of weighted average vs summary
        # e_average = np.average(e_chunks, axis=0, weights=[len(c) for c in chunks])
        e_summary = get_embedding(summary, config, self.embedding_provider)

        metadata["source_type"] = source_type

        return MemoryItem(
            raw_content=text,
            summary=summary,
            chunks=chunks,
            chunk_summaries=chunk_summaries,
            e_summary=e_summary,
            e_chunks=e_chunks,
            metadata=metadata,
        )

    def from_text_file(self, content: str, path: str, config: Config):
        return self.from_text(content, "text_file", config, {"location": path})

    def from_code_file(self, content: str, path: str):
        # TODO: implement tailored code memories
        return self.from_text(content, "code_file", {"location": path})

    def from_ai_action(self, ai_message: ChatMessage, result_message: ChatMessage):
        # The result_message contains either user feedback
        # or the result of the command specified in ai_message

        if ai_message.role != "assistant":
            raise ValueError(f"Invalid role on 'ai_message': {ai_message.role}")

        result = (
            result_message.content
            if result_message.content.startswith("Command")
            else "None"
        )
        user_input = (
            result_message.content
            if result_message.content.startswith("Human feedback")
            else "None"
        )
        memory_content = (
            f"Assistant Reply: {ai_message.content}"
            "\n\n"
            f"Result: {result}"
            "\n\n"
            f"Human Feedback: {user_input}"
        )

        return self.from_text(
            text=memory_content,
            source_type="agent_history",
            how_to_summarize=(
                "if possible, also make clear the link between the command in the"
                " assistant's response and the command result. "
                "Do not mention the human feedback if there is none.",
            ),
        )

    def from_webpage(
        self, content: str, url: str, config: Config, question: str | None = None
    ):
        return self.from_text(
            text=content,
            source_type="webpage",
            config=config,
            metadata={"location": url},
            question_for_summary=question,
        )


class MemoryItemRelevance(BaseModel):
    """
    Class that encapsulates memory relevance search functionality and data.
    Instances contain a MemoryItem and its relevance scores for a given query.
    """

    memory_item: MemoryItem
    for_query: str
    summary_relevance_score: float
    chunk_relevance_scores: list[float]

    @staticmethod
    def of(
        memory_item: MemoryItem, for_query: str, e_query: Embedding | None = None
    ) -> MemoryItemRelevance:
        e_query = e_query if e_query is not None else get_embedding(for_query)
        _, srs, crs = MemoryItemRelevance.calculate_scores(memory_item, e_query)
        return MemoryItemRelevance(
            for_query=for_query,
            memory_item=memory_item,
            summary_relevance_score=srs,
            chunk_relevance_scores=crs,
        )

    @staticmethod
    def calculate_scores(
        memory: MemoryItem, compare_to: Embedding
    ) -> tuple[float, float, list[float]]:
        """
        Calculates similarity between given embedding and all embeddings of the memory

        Returns:
            float: the aggregate (max) relevance score of the memory
            float: the relevance score of the memory summary
            list: the relevance scores of the memory chunks
        """
        summary_relevance_score = np.dot(memory.e_summary, compare_to)
        chunk_relevance_scores = np.dot(memory.e_chunks, compare_to).tolist()
        logger.debug(f"Relevance of summary: {summary_relevance_score}")
        logger.debug(f"Relevance of chunks: {chunk_relevance_scores}")

        relevance_scores = [summary_relevance_score, *chunk_relevance_scores]
        logger.debug(f"Relevance scores: {relevance_scores}")
        return max(relevance_scores), summary_relevance_score, chunk_relevance_scores

    @property
    def score(self) -> float:
        """The aggregate relevance score of the memory item for the given query"""
        return max([self.summary_relevance_score, *self.chunk_relevance_scores])

    @property
    def most_relevant_chunk(self) -> tuple[str, float]:
        """The most relevant chunk of the memory item + its score for the given query"""
        i_relmax = np.argmax(self.chunk_relevance_scores)
        return self.memory_item.chunks[i_relmax], self.chunk_relevance_scores[i_relmax]

    def __str__(self):
        return (
            f"{self.memory_item.summary} ({self.summary_relevance_score}) "
            f"{self.chunk_relevance_scores}"
        )
