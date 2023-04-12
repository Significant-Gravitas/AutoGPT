import asyncio
import hashlib
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup
from config import Config
from llm_utils import async_chat_completion, create_chat_completion, ChatRequest
import logging
cfg = Config()

# Define and check for local file address prefixes
def check_local_file_access(url):
    local_prefixes = ['file:///', 'file://localhost', 'http://localhost', 'https://localhost']
    return any(url.startswith(prefix) for prefix in local_prefixes)

def scrape_text(url)->str|None:
    """Scrape text from a webpage"""
    # Most basic check if the URL is valid:
    if not url.startswith('http'):
        return "Error: Invalid URL"

    # Restrict access to local files
    if check_local_file_access(url):
        return "Error: Access to local files is restricted"

    try:
        response = requests.get(url, headers=cfg.user_agent_header)
    except requests.exceptions.RequestException as e:
        return "Error: " + str(e)

    # Check if the response contains an HTTP error
    if response.status_code >= 400:
        logging.warning(f'response status: {response.status_code} = {response.text}')
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


def extract_hyperlinks(soup):
    """Extract hyperlinks from a BeautifulSoup object"""
    hyperlinks = []
    for link in soup.find_all('a', href=True):
        hyperlinks.append((link.text, link['href']))
    return hyperlinks


def format_hyperlinks(hyperlinks):
    """Format hyperlinks into a list of strings"""
    formatted_links = []
    for link_text, link_url in hyperlinks:
        formatted_links.append(f"{link_text} ({link_url})")
    return formatted_links


def scrape_links(url):
    """Scrape links from a webpage"""
    response = requests.get(url, headers=cfg.user_agent_header)

    # Check if the response contains an HTTP error
    if response.status_code >= 400:
        return "error"

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup)

    return format_hyperlinks(hyperlinks)


def split_text(text, max_length=8192):
    """Split text into chunks of a maximum length"""
    paragraphs = text.split("\n")
    current_length = 0
    current_chunk = []

    for i, paragraph in enumerate(paragraphs):
        if current_length + len(paragraph) + 1 <= max_length:
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 1
        else:
            yield "\n".join(current_chunk)
            current_chunk = [paragraph]
            current_length = len(paragraph) + 1

    if current_chunk:
        yield "\n".join(current_chunk)


def create_message(chunk, question):
    """Create a message for the user to summarize a chunk of text"""
    return {
        "role": "user",
        "content": f"\"\"\"{chunk}\"\"\" Using the above text, please answer the following question: \"{question}\" -- if the question cannot be answered using the text, please summarize the text."
    }


@dataclass
class TextSummary:
    source: str
    summary: str


async def summarize_text(text, question, source: str | None = None, max_tokens=300) -> Optional[TextSummary]:
    """Summarize text using the LLM model"""
    if not text:
        logging.warning('No text to summarize')
        return None

    text_length = len(text)
    logging.debug(f"Text length: {text_length} characters")

    chunks = list(split_text(text))
    tasks = [
        async_chat_completion(
            request=ChatRequest.from_messages(messages=[create_message(chunk, question)], max_tokens=max_tokens, key=i))
        for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    def filter_result(result) -> bool:
        if isinstance(result, BaseException):
            logging.error(result)
            return False
        if result.exception:
            logging.error(f'{result.exception}')
            return False
        return True

    summaries = list(map(lambda c: c.reply, sorted(filter(lambda r: filter_result(r), results), key=lambda c: c.key())))
    logging.info(f"Summarized {len(chunks)} chunks.")

    combined_summary = "\n".join(summaries)
    messages = [create_message(combined_summary, question)]
    final_summary = await async_chat_completion(ChatRequest.from_messages(messages=messages,max_tokens=max_tokens))
    if final_summary.reply:
        source = source or hashlib.md5(text).hexdigest()
        return TextSummary(source=source, summary=final_summary.reply)
    else:
        return None


