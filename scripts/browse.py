import requests
from bs4 import BeautifulSoup
from config import Config
from llm_utils import create_chat_completion

cfg = Config()

def scrape_text(url):
    response = requests.get(url)

    # Check if the response contains an HTTP error
    if response.status_code >= 400:
        return "Error: HTTP " + str(response.status_code) + " error"

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


def extract_hyperlinks(soup):
    hyperlinks = []
    for link in soup.find_all('a', href=True):
        hyperlinks.append((link.text, link['href']))
    return hyperlinks


def format_hyperlinks(hyperlinks):
    formatted_links = []
    for link_text, link_url in hyperlinks:
        formatted_links.append(f"{link_text} ({link_url})")
    return formatted_links


def scrape_links(url):
    response = requests.get(url)

    # Check if the response contains an HTTP error
    if response.status_code >= 400:
        return "error"

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup)

    return format_hyperlinks(hyperlinks)


def split_text(text, max_length=8192):
    paragraphs = text.split("\n")
    current_length = 0
    current_chunk = []

    for paragraph in paragraphs:
        if current_length + len(paragraph) + 1 <= max_length:
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 1
        else:
            yield "\n".join(current_chunk)
            current_chunk = [paragraph]
            current_length = len(paragraph) + 1

    if current_chunk:
        yield "\n".join(current_chunk)


def summarize_text(text, is_website=True):
    if text == "":
        return "Error: No text to summarize"

    print("Text length: " + str(len(text)) + " characters")
    summaries = []
    chunks = list(split_text(text))

    for i, chunk in enumerate(chunks):
        print("Summarizing chunk " + str(i + 1) + " / " + str(len(chunks)))
        if is_website:
            messages = [
                {
                    "role": "user",
                    "content": "Please summarize the following website text, do not describe the general website, but instead concisely extract the specific information this subpage contains.: " +
                    chunk},
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": "Please summarize the following text, focusing on extracting concise and specific information: " +
                    chunk},
            ]

        summary = create_chat_completion(
            model=cfg.fast_llm_model,
            messages=messages,
            max_tokens=300,
        )
        summaries.append(summary)
    print("Summarized " + str(len(chunks)) + " chunks.")

    combined_summary = "\n".join(summaries)

    # Summarize the combined summary
    if is_website:
        messages = [
            {
                "role": "user",
                "content": "Please summarize the following website text, do not describe the general website, but instead concisely extract the specific information this subpage contains.: " +
                combined_summary},
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": "Please summarize the following text, focusing on extracting concise and specific infomation: " +
                combined_summary},
        ]

    final_summary = create_chat_completion(
        model=cfg.fast_llm_model,
        messages=messages,
        max_tokens=300,
    )

    return final_summary
