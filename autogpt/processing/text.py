"""Text processing functions"""
from math import ceil
from typing import Optional

import spacy
import tiktoken

from autogpt.config import Config
from autogpt.llm.base import ChatSequence
from autogpt.llm.providers.openai import OPEN_AI_MODELS
from autogpt.llm.utils import count_string_tokens, create_chat_completion
from autogpt.logs import logger
from autogpt.utils import batch

CFG = Config()


def _max_chunk_length(model: str, max: Optional[int] = None) -> int:
    model_max_input_tokens = OPEN_AI_MODELS[model].max_tokens - 1
    if max is not None and max > 0:
        return min(max, model_max_input_tokens)
    return model_max_input_tokens


def must_chunk_content(
    text: str, for_model: str, max_chunk_length: Optional[int] = None
) -> bool:
    return count_string_tokens(text, for_model) > _max_chunk_length(
        for_model, max_chunk_length
    )


def chunk_content(
    content: str,
    for_model: str,
    max_chunk_length: Optional[int] = None,
    with_overlap=True,
):
    """Split content into chunks of approximately equal token length."""

    MAX_OVERLAP = 200  # limit overlap to save tokens

    if not must_chunk_content(content, for_model, max_chunk_length):
        yield content, count_string_tokens(content, for_model)
        return

    max_chunk_length = max_chunk_length or _max_chunk_length(for_model)

    tokenizer = tiktoken.encoding_for_model(for_model)

    tokenized_text = tokenizer.encode(content)
    total_length = len(tokenized_text)
    n_chunks = ceil(total_length / max_chunk_length)

    chunk_length = ceil(total_length / n_chunks)
    overlap = min(max_chunk_length - chunk_length, MAX_OVERLAP) if with_overlap else 0

    for token_batch in batch(tokenized_text, chunk_length + overlap, overlap):
        yield tokenizer.decode(token_batch), len(token_batch)


def summarize_text(
    text: str, instruction: Optional[str] = None, question: Optional[str] = None
) -> tuple[str, None | list[tuple[str, str]]]:
    """Summarize text using the OpenAI API

    Args:
        text (str): The text to summarize
        instruction (str): Additional instruction for summarization, e.g. "focus on information related to polar bears", "omit personal information contained in the text"

    Returns:
        str: The summary of the text
        list[(summary, chunk)]: Text chunks and their summary, if the text was chunked.
            None otherwise.
    """
    if not text:
        raise ValueError("No text to summarize")

    if instruction and question:
        raise ValueError("Parameters 'question' and 'instructions' cannot both be set")

    model = CFG.fast_llm_model

    if question:
        instruction = (
            f'include any information that can be used to answer the question "{question}". '
            "Do not directly answer the question itself"
        )

    summarization_prompt = ChatSequence.for_model(model)

    token_length = count_string_tokens(text, model)
    logger.info(f"Text length: {token_length} tokens")

    # reserve 50 tokens for summary prompt, 500 for the response
    max_chunk_length = _max_chunk_length(model) - 550
    logger.info(f"Max chunk length: {max_chunk_length} tokens")

    if not must_chunk_content(text, model, max_chunk_length):
        # summarization_prompt.add("user", text)
        summarization_prompt.add(
            "user",
            "Write a concise summary of the following text"
            f"{f'; {instruction}' if instruction is not None else ''}:"
            "\n\n\n"
            f'LITERAL TEXT: """{text}"""'
            "\n\n\n"
            "CONCISE SUMMARY: The text is best summarized as"
            # "Only respond with a concise summary or description of the user message."
        )

        logger.debug(f"Summarizing with {model}:\n{summarization_prompt.dump()}\n")
        summary = create_chat_completion(
            summarization_prompt, temperature=0, max_tokens=500
        )

        logger.debug(f"\n{'-'*16} SUMMARY {'-'*17}\n{summary}\n{'-'*42}\n")
        return summary.strip(), None

    summaries: list[str] = []
    chunks = list(split_text(text, for_model=model, max_chunk_length=max_chunk_length))

    for i, (chunk, chunk_length) in enumerate(chunks):
        logger.info(
            f"Summarizing chunk {i + 1} / {len(chunks)} of length {chunk_length} tokens"
        )
        summary, _ = summarize_text(chunk, instruction)
        summaries.append(summary)

    logger.info(f"Summarized {len(chunks)} chunks")

    summary, _ = summarize_text("\n\n".join(summaries))

    return summary.strip(), [
        (summaries[i], chunks[i][0]) for i in range(0, len(chunks))
    ]


def split_text(
    text: str,
    for_model: str = CFG.fast_llm_model,
    with_overlap=True,
    max_chunk_length: Optional[int] = None,
):
    """Split text into chunks of sentences, with each chunk not exceeding the maximum length

    Args:
        text (str): The text to split
        for_model (str): The model to chunk for; determines tokenizer and constraints
        max_length (int, optional): The maximum length of each chunk

    Yields:
        str: The next chunk of text

    Raises:
        ValueError: when a sentence is longer than the maximum length
    """
    max_length = _max_chunk_length(for_model, max_chunk_length)

    # flatten paragraphs to improve performance
    text = text.replace("\n", " ")
    text_length = count_string_tokens(text, for_model)

    if text_length < max_length:
        yield text, text_length
        return

    n_chunks = ceil(text_length / max_length)
    target_chunk_length = ceil(text_length / n_chunks)

    nlp: spacy.language.Language = spacy.load(CFG.browse_spacy_language_model)
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
        sentence_length = count_string_tokens(sentence, for_model)
        expected_chunk_length = current_chunk_length + 1 + sentence_length

        if (
            expected_chunk_length < max_length
            # try to create chunks of approximately equal size
            and expected_chunk_length - (sentence_length / 2) < target_chunk_length
        ):
            current_chunk.append(sentence)
            current_chunk_length = expected_chunk_length

        elif sentence_length < max_length:
            if last_sentence:
                yield " ".join(current_chunk), current_chunk_length
                current_chunk = []
                current_chunk_length = 0

                if with_overlap:
                    overlap_max_length = max_length - sentence_length - 1
                    if last_sentence_length < overlap_max_length:
                        current_chunk += [last_sentence]
                        current_chunk_length += last_sentence_length + 1
                    elif overlap_max_length > 5:
                        # add as much from the end of the last sentence as fits
                        current_chunk += [
                            list(
                                chunk_content(
                                    last_sentence,
                                    for_model,
                                    overlap_max_length,
                                )
                            ).pop()[0],
                        ]
                        current_chunk_length += overlap_max_length + 1

            current_chunk += [sentence]
            current_chunk_length += sentence_length

        else:  # sentence longer than maximum length -> chop up and try again
            sentences[i : i + 1] = [
                chunk
                for chunk, _ in chunk_content(sentence, for_model, target_chunk_length)
            ]
            continue

        i += 1
        last_sentence = sentence
        last_sentence_length = sentence_length

    if current_chunk:
        yield " ".join(current_chunk), current_chunk_length
