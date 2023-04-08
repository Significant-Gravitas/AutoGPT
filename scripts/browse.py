import requests
from bs4 import BeautifulSoup
from config import Config
from llm_utils import create_chat_completion

cfg = Config()

# Fetch URL and return error message (if any) and parsed HTML content
def fetch_url(url):
    response = requests.get(url, headers=cfg.user_agent_header)

    if response.status_code >= 400:
        return "Error: HTTP " + str(response.status_code) + " error", None

    return None, BeautifulSoup(response.text, "html.parser")

# Remove script and style tags from the parsed HTML content
def remove_scripts_and_styles(soup):
    for element in soup(["script", "style"]):
        element.extract()

# Scrape text content from a given URL
def scrape_text(url):
    error, soup = fetch_url(url)

    if error:
        return error

    remove_scripts_and_styles(soup)

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

# Extract and return hyperlinks (text and URL) from parsed HTML content
def extract_hyperlinks(soup):
    return [(link.text, link['href']) for link in soup.find_all('a', href=True)]

# Format a list of hyperlinks as strings with text and URL
def format_hyperlinks(hyperlinks):
    return [f"{link_text} ({link_url})" for link_text, link_url in hyperlinks]

# Scrape hyperlinks from a given URL and return them formatted as strings
def scrape_links(url):
    error, soup = fetch_url(url)

    if error:
        return "error"

    remove_scripts_and_styles(soup)

    hyperlinks = extract_hyperlinks(soup)
    return format_hyperlinks(hyperlinks)

# Split text into chunks based on a specified maximum length
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

# Create a message object for LLM with text chunk and question
def create_message(chunk, question):
    return {
        "role": "user",
        "content": f"\"\"\"{chunk}\"\"\" Using the above text, please answer the following question: \"{question}\" -- if the question cannot be answered using the text, please summarize the text."
    }

# Summarize text using LLM and a given question
def summarize_text(text, question):
    if not text:
        return "Error: No text to summarize"

    text_length = len(text)
    print(f"Text length: {text_length} characters")

    summaries = []
    chunks = list(split_text(text))

    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1} / {len(chunks)}")
        messages = [create_message(chunk, question)]

        summary = create_chat_completion(
            model=cfg.fast_llm_model,
            messages=messages,
            max_tokens=300,
        )
        summaries.append(summary)

    print(f"Summarized {len(chunks)} chunks.")

    combined_summary = "\n".join(summaries)
    messages = [create_message(combined_summary, question)]

    final_summary = create_chat_completion(
        model=cfg.fast_llm_model,
        messages=messages,
        max_tokens=300,
        )

    return final_summary
