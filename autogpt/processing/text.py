"""Text processing functions"""
from typing import Dict, Generator, Optional

import spacy
from selenium.webdriver.remote.webdriver import WebDriver

from autogpt import token_counter
from autogpt.config import Config
from autogpt.llm_utils import create_chat_completion
from autogpt.memory import get_memory

CFG = Config()


def split_text(
    text: str,
    max_length: int = CFG.browse_chunk_max_length,
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
            token_usage_of_chunk(messages=message_with_additional_sentence, model=model)
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
                token_usage_of_chunk(messages=message_this_sentence_only, model=model)
                + 1
            )
            if expected_token_usage > max_length:
                raise ValueError(
                    f"Sentence is too long in webpage: {expected_token_usage} tokens."
                )

    if current_chunk:
        yield " ".join(current_chunk)


def token_usage_of_chunk(messages, model):
    return token_counter.count_message_tokens(messages, model)


def summarize_text(
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
    print(f"Text length: {text_length} characters")

    summaries = []
    chunks = list(
        split_text(
            text, max_length=CFG.browse_chunk_max_length, model=model, question=question
        ),
    )
    scroll_ratio = 1 / len(chunks)

    for i, chunk in enumerate(chunks):
        if driver:
            scroll_to_percentage(driver, scroll_ratio * i)
        print(f"Adding chunk {i + 1} / {len(chunks)} to memory")

        memory_to_add = f"Source: {url}\n" f"Raw content part#{i + 1}: {chunk}"

        memory = get_memory(CFG)
        memory.add(memory_to_add)

        messages = [create_message(chunk, question)]
        tokens_for_chunk = token_counter.count_message_tokens(messages, model)
        print(
            f"Summarizing chunk {i + 1} / {len(chunks)} of length {len(chunk)} characters, or {tokens_for_chunk} tokens"
        )

        summary = create_chat_completion(
            model=model,
            messages=messages,
        )
        summaries.append(summary)
        print(
            f"Added chunk {i + 1} summary to memory, of length {len(summary)} characters"
        )

        memory_to_add = f"Source: {url}\n" f"Content summary part#{i + 1}: {summary}"

        memory.add(memory_to_add)

    print(f"Summarized {len(chunks)} chunks.")

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
