"""Text processing functions"""
from math import ceil
from typing import Dict, Generator, Optional

import spacy
import tiktoken
from selenium.webdriver.remote.webdriver import WebDriver

from autogpt.config import Config
from autogpt.llm import count_message_tokens, create_chat_completion
from autogpt.llm.llm_utils import create_text_completion
from autogpt.llm.providers.openai import OPEN_AI_MODELS
from autogpt.llm.token_counter import count_string_tokens
from autogpt.logs import logger
from autogpt.memory import get_memory
from autogpt.utils import batch

CFG = Config()


def _max_chunk_length(model: str, max: int = None):
    model_max_input_tokens = OPEN_AI_MODELS[model].max_tokens - 1
    if max is not None and max > 0:
        return min(max, model_max_input_tokens)
    return model_max_input_tokens


def must_chunk_text(text: str, for_model: str, max_chunk_length: int = None):
    return count_string_tokens(text) > _max_chunk_length(for_model, max_chunk_length)


def chunk_text(
    text: str, for_model: str, max_chunk_length: int = None, with_overlap=True
):
    if not must_chunk_text(text, for_model, max_chunk_length):
        yield text, count_string_tokens(text, for_model)
        return

    tokenizer = tiktoken.encoding_for_model(for_model)

    tokenized_text = tokenizer.encode(text)
    total_length = len(tokenized_text)
    n_chunks = ceil(total_length / max_chunk_length)

    chunk_length = ceil(total_length / n_chunks)
    if with_overlap:
        overlap = min(max_chunk_length - chunk_length, max_chunk_length / 4)
    else:
        overlap = 0

    for token_batch in batch(tokenized_text, chunk_length + overlap, overlap):
        yield tokenizer.decode(token_batch), len(token_batch)


def summarize_text(text: str, question: Optional[str]) -> str:
    """Summarize text using the OpenAI API

    Args:
        text (str): The text to summarize
        question (str): A question to focus the summary content

    Returns:
        str: The summary of the text
    """
    if not text:
        return "Error: No text to summarize"

    model = CFG.fast_llm_model

    summarization_prompt_template = (
        (
            "Describe and summarize the following document, focusing on information"
            f'related to the question "{question}".'
            "Do not answer the question itself.\n"
            '\nDocument: """{content}"""\n'
            "\nSummary/description:"
        )
        if question is not None
        else (
            "Describe and summarize the following document, "
            "covering all major topics present in the source text.\n"
            '\nDocument: """{content}"""\n'
            "\nSummary/description:"
        )
    )

    token_length = count_string_tokens(text, model)
    logger.info(f"Text length: {token_length} tokens")

    if not must_chunk_text(
        text, model, _max_chunk_length(model) - 550
    ):  # reserve 50 tokens for summary prompt, 500 for the response
        prompt = summarization_prompt_template.format(content=text)
        logger.debug(f"Summarizing with {model}:\n{'-'*32}\n{prompt}\n{'-'*32}\n")
        return create_text_completion(
            prompt, model, temperature=0, max_output_tokens=500
        )

    summaries: list[str] = []
    chunks = chunk_text(text, for_model=model)

    for i, (chunk, chunk_length) in enumerate(chunks):
        logger.info(
            f"Summarizing chunk {i + 1} / {len(chunks)} of length {chunk_length} tokens"
        )
        summary = summarize_text(chunk, question)
        summaries.append(summary)

    logger.info(f"Summarized {len(chunks)} chunks")

    return summarize_text("\n\n".join(summaries))


def split_text(
    text: str,
    model: str = CFG.fast_llm_model,
    question: str = "",
) -> Generator[str, None, None]:
    """Split text into chunks of a maximum length

    Args:
        text (str): The text to split
        max_length (int, optional): The maximum length of each chunk. Defaults to 8192.

    Yields:
        str: The next chunk of text

    Raises:
        ValueError: If the text is longer than the maximum length
    """
    max_length = round(_max_chunk_length(CFG.embedding_model) * 0.75)

    flatened_paragraphs = " ".join(text.split("\n"))
    nlp = spacy.load(CFG.browse_spacy_language_model)
    nlp.add_pipe("sentencizer")
    doc = nlp(flatened_paragraphs)
    sentences = [sent.text.strip() for sent in doc.sents]

    current_chunk = []

    for sentence in sentences:
        message_with_additional_sentence = [
            create_message(" ".join(current_chunk) + " " + sentence, question)
        ]

        expected_token_usage = (
            count_message_tokens(messages=message_with_additional_sentence, model=model)
            + 1
        )
        if expected_token_usage <= max_length:
            current_chunk.append(sentence)
        else:
            yield " ".join(current_chunk)
            current_chunk = [sentence]
            message_this_sentence_only = [
                create_message(" ".join(current_chunk), question)
            ]
            expected_token_usage = (
                count_message_tokens(messages=message_this_sentence_only, model=model)
                + 1
            )
            if expected_token_usage > max_length:
                raise ValueError(
                    f"Sentence is too long in webpage: {expected_token_usage} tokens."
                )

    if current_chunk:
        yield " ".join(current_chunk)


def summarize_memorize_webpage(
    url: str, text: str, question: str, driver: Optional[WebDriver] = None
) -> str:
    """Summarize text using the OpenAI API

    Args:
        url (str): The url of the text
        text (str): The text to summarize
        question (str): The question to ask the model
        driver (WebDriver): The webdriver to use to scroll the page

    Returns:
        str: The summary of the text
    """
    if not text:
        return "Error: No text to summarize"

    model = CFG.fast_llm_model
    text_length = len(text)
    logger.info(f"Text length: {text_length} characters")

    summaries = []
    chunks = list(
        split_text(text, model=model, question=question),
    )
    scroll_ratio = 1 / len(chunks)

    for i, chunk in enumerate(chunks):
        if driver:
            scroll_to_percentage(driver, scroll_ratio * i)
        logger.info(f"Adding chunk {i + 1} / {len(chunks)} to memory")

        memory_to_add = f"Source: {url}\n" f"Raw content part#{i + 1}: {chunk}"

        memory = get_memory(CFG)
        memory.add(memory_to_add)

        messages = [create_message(chunk, question)]
        tokens_for_chunk = count_message_tokens(messages, model)
        logger.info(
            f"Summarizing chunk {i + 1} / {len(chunks)} of length {len(chunk)} characters, or {tokens_for_chunk} tokens"
        )

        summary = create_chat_completion(
            model=model,
            messages=messages,
        )
        summaries.append(summary)
        logger.info(
            f"Added chunk {i + 1} summary to memory, of length {len(summary)} characters"
        )

        memory_to_add = f"Source: {url}\n" f"Content summary part#{i + 1}: {summary}"

        memory.add(memory_to_add)

    logger.info(f"Summarized {len(chunks)} chunks.")

    combined_summary = "\n".join(summaries)
    messages = [create_message(combined_summary, question)]

    return create_chat_completion(
        model=model,
        messages=messages,
    )


def scroll_to_percentage(driver: WebDriver, ratio: float) -> None:
    """Scroll to a percentage of the page

    Args:
        driver (WebDriver): The webdriver to use
        ratio (float): The percentage to scroll to

    Raises:
        ValueError: If the ratio is not between 0 and 1
    """
    if ratio < 0 or ratio > 1:
        raise ValueError("Percentage should be between 0 and 1")
    driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight * {ratio});")


def create_message(chunk: str, question: str) -> Dict[str, str]:
    """Create a message for the chat completion

    Args:
        chunk (str): The chunk of text to summarize
        question (str): The question to answer

    Returns:
        Dict[str, str]: The message to send to the chat completion
    """
    return {
        "role": "user",
        "content": f'"""{chunk}""" Using the above text, answer the following'
        f' question: "{question}" -- if the question cannot be answered using the text,'
        " summarize the text.",
    }
