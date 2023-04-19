"""Text processing functions"""
from typing import Dict, Generator, Optional, List
from selenium.webdriver.remote.webdriver import WebDriver

from autogpt.config import Config
from autogpt.llm_utils import create_chat_completion
from autogpt.memory import get_memory

CFG = Config()

def get_memory_instance():
    return get_memory(CFG)

def split_text(text: str, max_length: int = 50) -> Generator[str, None, None]:
    if max_length <= 0:
        raise ValueError("max_length should be greater than 0")

    if not text:
        return

    lines = text.split("\n")

    for line in lines:
        words = line.split()
        current_line = []

        for word in words:
            if len(" ".join(current_line)) + len(word) + 1 > max_length:
                yield " ".join(current_line)
                current_line = [word]
            else:
                current_line.append(word)

        if current_line:
            yield " ".join(current_line)



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
    MEMORY = get_memory_instance()

    text_length = len(text)
    print(f"Text length: {text_length} characters")

    summaries = []
    chunks = list(split_text(text))
    scroll_ratio = 1 / len(chunks)

    for i, chunk in enumerate(chunks):
        if driver:
            scroll_to_percentage(driver, scroll_ratio * i)
        print(f"Adding chunk {i + 1} / {len(chunks)} to memory")

        memory_to_add = f"Source: {url}\n" f"Raw content part#{i + 1}: {chunk}"

        MEMORY.add(memory_to_add)

        print(f"Summarizing chunk {i + 1} / {len(chunks)}")
        messages = [create_message(chunk, question)]

        summary = create_chat_completion(
            model=CFG.fast_llm_model,
            messages=messages,
        )
        summaries.append(summary)
        print(f"Added chunk {i + 1} summary to memory")

        memory_to_add = f"Source: {url}\n" f"Content summary part#{i + 1}: {summary}"

        MEMORY.add(memory_to_add)

    print(f"Summarized {len(chunks)} chunks.")

    combined_summary = "\n".join(summaries)
    messages = [create_message(combined_summary, question)]

    return create_chat_completion(
        model=CFG.fast_llm_model,
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
