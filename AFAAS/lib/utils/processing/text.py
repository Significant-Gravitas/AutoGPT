"""Text processing functions"""

import json
import logging
import math
import os
from typing import Iterator, Optional, TypeVar

import spacy

from langchain_core.messages  import AIMessage , HumanMessage, SystemMessage , ChatMessage
from AFAAS.interfaces.adapters import AbstractChatModelProvider
from AFAAS.interfaces.adapters.chatmodel import ChatPrompt
from AFAAS.interfaces.adapters.language_model import ModelTokenizer
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)
LOG.notice(
    f"Looking for volunteer to migrate {os.path.relpath(__file__)} library to Langchain."
)

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
    llm_provider: AbstractChatModelProvider,
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

    model = "gpt-3.5-turbo"

    if question:
        if instruction:
            raise ValueError(
                "Parameters 'question' and 'instructions' cannot both be set"
            )

        instruction = (
            f'From the text, answer the question: "{question}". '
            "If the answer is not in the text, indicate this clearly "
            "and concisely state why the text is not suitable to answer the question."
        )
    elif not instruction:
        instruction = (
            "Summarize or describe the text clearly and concisely, "
            "whichever seems more appropriate."
        )

    return await _process_text(  # type: ignore
        text=text,
        instruction=instruction,
        llm_provider=llm_provider,
    )


async def extract_information(
    source_text: str,
    topics_of_interest: list[str],
    llm_provider: AbstractChatModelProvider,
) -> list[str]:
    fmt_topics_list = "\n".join(f"* {topic}." for topic in topics_of_interest)
    instruction = (
        "Extract relevant pieces of information about the following topics:\n"
        f"{fmt_topics_list}\n"
        "Reword pieces of information if needed to make them self-explanatory. "
        "Be concise.\n\n"
        "Respond with an `Array<string>` in JSON format AND NOTHING ELSE. "
        'If the text contains no relevant information, return "[]".'
    )
    return await _process_text(  # type: ignore
        text=source_text,
        instruction=instruction,
        output_type=list[str],
        llm_provider=llm_provider,
    )


async def _process_text(
    text: str,
    instruction: str,
    llm_provider: AbstractChatModelProvider,
    output_type: type[str | list[str]] = str,
) -> tuple[str, list[tuple[str, str]]] | list[str]:
    """Process text using the OpenAI API for summarization or information extraction

    Params:
        text (str): The text to process.
        instruction (str): Additional instruction for processing.
        llm_provider: LLM provider to use.
        config (Config): The global application config.
        output_type: `str` for summaries or `list[str]` for piece-wise info extraction.

    Returns:
        For summarization: tuple[str, None | list[(summary, chunk)]]
        For piece-wise information extraction: list[str]
    """
    if not text.strip():
        raise ValueError("No content")

    model = "gpt-3"

    text_tlength = llm_provider.count_tokens(text, model)
    LOG.debug(f"Text length: {text_tlength} tokens")

    max_result_tokens = 500
    max_chunk_length = llm_provider.get_token_limit(model) - max_result_tokens - 50
    LOG.debug(f"Max chunk length: {max_chunk_length} tokens")

    if text_tlength < max_chunk_length:
        prompt = ChatPrompt(
            messages=[
                SystemMessage(
                    "The user is going to give you a text enclosed in triple quotes. "
                    f"{instruction}"
                ),
                HumanMessage(f'"""{text}"""'),
            ]
        )

        LOG.debug(f"PROCESSING:\n{prompt}")

        response = await llm_provider.create_chat_completion(
            model_prompt=prompt.messages,
            llm_model_name=model,
            temperature=0.5,
            max_tokens=max_result_tokens,
            completion_parser=lambda s: (
                json.loads(s.content) if output_type is not str else None
            ),
        )

        if output_type == list[str]:
            LOG.debug(f"Raw LLM response: {repr(response.response.content)}")
            fmt_result_bullet_list = "\n".join(f"* {r}" for r in response.parsed_result)
            LOG.debug(
                f"\n{'-'*11} EXTRACTION RESULT {'-'*12}\n"
                f"{fmt_result_bullet_list}\n"
                f"{'-'*42}\n"
            )
            return response.parsed_result
        else:
            summary = response.response.content
            LOG.debug(f"\n{'-'*16} SUMMARY {'-'*17}\n{summary}\n{'-'*42}\n")
            return summary.strip(), [(summary, text)]
    else:
        chunks = list(
            split_text(
                text,
                max_chunk_length=max_chunk_length,
                tokenizer=llm_provider.get_tokenizer(model),
            )
        )

        processed_results = []
        for i, (chunk, _) in enumerate(chunks):
            LOG.info(f"Processing chunk {i + 1} / {len(chunks)}")
            chunk_result = await _process_text(
                text=chunk,
                instruction=instruction,
                output_type=output_type,
                llm_provider=llm_provider,
            )
            processed_results.extend(
                chunk_result if output_type == list[str] else [chunk_result]
            )

        if output_type == list[str]:
            return processed_results
        else:
            summary, _ = await _process_text(
                "\n\n".join([result[0] for result in processed_results]),
                instruction=(
                    "The text consists of multiple partial summaries. "
                    "Combine these partial summaries into one."
                ),
                llm_provider=llm_provider,
            )
            return summary.strip(), [
                (processed_results[i], chunks[i][0]) for i in range(0, len(chunks))
            ]


def split_text(
    text: str,
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

    nlp: spacy.language.Language = spacy.load("en_core_web_sm")
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
