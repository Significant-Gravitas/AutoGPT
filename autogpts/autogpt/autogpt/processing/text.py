"""Text processing functions"""
import logging
import math
from typing import Iterator, Optional, TypeVar

import spacy

from autogpt.config import Config
from autogpt.core.prompting import ChatPrompt
from autogpt.core.resource.model_providers import (
    ChatMessage,
    ChatModelProvider,
    ModelTokenizer,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def batch(
    sequence: list[T], max_batch_length: int, overlap: int = 0
) -> Iterator[list[T]]:
    """
    Batch data from iterable into slices of length N. The last batch may be shorter.

    Example: `batched('ABCDEFGHIJ', 3)` --> `ABC DEF GHI J`
    """
    if max_batch_length < 1:
        raise ValueError("n must be at least one")
    for i in range(0, len(sequence), max_batch_length - overlap):
        yield sequence[i : i + max_batch_length]


def chunk_content(
    content: str,
    max_chunk_length: int,
    tokenizer: ModelTokenizer,
    with_overlap: bool = True,
) -> Iterator[tuple[str, int]]:
    """Split content into chunks of approximately equal token length."""

    MAX_OVERLAP = 200  # limit overlap to save tokens

    tokenized_text = tokenizer.encode(content)
    total_length = len(tokenized_text)
    n_chunks = math.ceil(total_length / max_chunk_length)

    chunk_length = math.ceil(total_length / n_chunks)
    overlap = min(max_chunk_length - chunk_length, MAX_OVERLAP) if with_overlap else 0

    for token_batch in batch(tokenized_text, chunk_length + overlap, overlap):
        yield tokenizer.decode(token_batch), len(token_batch)


async def summarize_text(
    text: str,
    llm_provider: ChatModelProvider,
    config: Config,
    instruction: Optional[str] = None,
    question: Optional[str] = None,
) -> tuple[str, None | list[tuple[str, str]]]:
    """Summarize text using the OpenAI API

    Args:
        text (str): The text to summarize.
        llm_provider: LLM provider to use for summarization.
        config (Config): The global application config, containing the FAST_LLM setting.
        instruction (str): Additional instruction for summarization, e.g.
            "focus on information related to polar bears", or
            "omit personal information contained in the text".
        question (str): Question to be answered by the summary.

    Returns:
        str: The summary of the text
        list[(summary, chunk)]: Text chunks and their summary, if the text was chunked.
            None otherwise.
    """
    if not text:
        raise ValueError("No text to summarize")

    if instruction and question:
        raise ValueError("Parameters 'question' and 'instructions' cannot both be set")

    model = config.fast_llm

    if question:
        instruction = (
            'Include any information that can be used to answer the question: "%s". '
            "Do not directly answer the question itself."
        ) % question

    summarization_prompt = ChatPrompt(messages=[])

    text_tlength = llm_provider.count_tokens(text, model)
    logger.info(f"Text length: {text_tlength} tokens")

    # reserve 50 tokens for summary prompt, 500 for the response
    max_chunk_length = llm_provider.get_token_limit(model) - 550
    logger.info(f"Max chunk length: {max_chunk_length} tokens")

    if text_tlength < max_chunk_length:
        # summarization_prompt.add("user", text)
        summarization_prompt.messages.append(
            ChatMessage.user(
                "Write a concise summary of the following text."
                f"{f' {instruction}' if instruction is not None else ''}:"
                "\n\n\n"
                f'LITERAL TEXT: """{text}"""'
                "\n\n\n"
                "CONCISE SUMMARY: The text is best summarized as"
            )
        )

        summary = (
            await llm_provider.create_chat_completion(
                model_prompt=summarization_prompt.messages,
                model_name=model,
                temperature=0,
                max_tokens=500,
            )
        ).response["content"]

        logger.debug(f"\n{'-'*16} SUMMARY {'-'*17}\n{summary}\n{'-'*42}\n")
        return summary.strip(), None

    summaries: list[str] = []
    chunks = list(
        split_text(
            text,
            config=config,
            max_chunk_length=max_chunk_length,
            tokenizer=llm_provider.get_tokenizer(model),
        )
    )

    for i, (chunk, chunk_length) in enumerate(chunks):
        logger.info(
            f"Summarizing chunk {i + 1} / {len(chunks)} of length {chunk_length} tokens"
        )
        summary, _ = await summarize_text(
            text=chunk,
            instruction=instruction,
            llm_provider=llm_provider,
            config=config,
        )
        summaries.append(summary)

    logger.info(f"Summarized {len(chunks)} chunks")

    summary, _ = await summarize_text(
        "\n\n".join(summaries),
        llm_provider=llm_provider,
        config=config,
    )
    return summary.strip(), [
        (summaries[i], chunks[i][0]) for i in range(0, len(chunks))
    ]


def split_text(
    text: str,
    config: Config,
    max_chunk_length: int,
    tokenizer: ModelTokenizer,
    with_overlap: bool = True,
) -> Iterator[tuple[str, int]]:
    """
    Split text into chunks of sentences, with each chunk not exceeding the max length.

    Args:
        text (str): The text to split.
        config (Config): Config object containing the Spacy model setting.
        max_chunk_length (int, optional): The maximum length of a chunk.
        tokenizer (ModelTokenizer): Tokenizer to use for determining chunk length.
        with_overlap (bool, optional): Whether to allow overlap between chunks.

    Yields:
        str: The next chunk of text

    Raises:
        ValueError: when a sentence is longer than the maximum length
    """
    text_length = len(tokenizer.encode(text))

    if text_length < max_chunk_length:
        yield text, text_length
        return

    n_chunks = math.ceil(text_length / max_chunk_length)
    target_chunk_length = math.ceil(text_length / n_chunks)

    nlp: spacy.language.Language = spacy.load(config.browse_spacy_language_model)
    nlp.add_pipe("sentencizer")
    doc = nlp(text)
    sentences = [sentence.text.strip() for sentence in doc.sents]

    current_chunk: list[str] = []
    current_chunk_length = 0
    last_sentence = None
    last_sentence_length = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_length = len(tokenizer.encode(sentence))
        expected_chunk_length = current_chunk_length + 1 + sentence_length

        if (
            expected_chunk_length < max_chunk_length
            # try to create chunks of approximately equal size
            and expected_chunk_length - (sentence_length / 2) < target_chunk_length
        ):
            current_chunk.append(sentence)
            current_chunk_length = expected_chunk_length

        elif sentence_length < max_chunk_length:
            if last_sentence:
                yield " ".join(current_chunk), current_chunk_length
                current_chunk = []
                current_chunk_length = 0

                if with_overlap:
                    overlap_max_length = max_chunk_length - sentence_length - 1
                    if last_sentence_length < overlap_max_length:
                        current_chunk += [last_sentence]
                        current_chunk_length += last_sentence_length + 1
                    elif overlap_max_length > 5:
                        # add as much from the end of the last sentence as fits
                        current_chunk += [
                            list(
                                chunk_content(
                                    content=last_sentence,
                                    max_chunk_length=overlap_max_length,
                                    tokenizer=tokenizer,
                                )
                            ).pop()[0],
                        ]
                        current_chunk_length += overlap_max_length + 1

            current_chunk += [sentence]
            current_chunk_length += sentence_length

        else:  # sentence longer than maximum length -> chop up and try again
            sentences[i : i + 1] = [
                chunk
                for chunk, _ in chunk_content(sentence, target_chunk_length, tokenizer)
            ]
            continue

        i += 1
        last_sentence = sentence
        last_sentence_length = sentence_length

    if current_chunk:
        yield " ".join(current_chunk), current_chunk_length
