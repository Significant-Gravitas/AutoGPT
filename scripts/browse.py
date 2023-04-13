import requests
from bs4 import BeautifulSoup
from config import Config
from llm_utils import create_chat_completion
from urllib.parse import urlparse, urljoin

cfg = Config()


# Function to check if the URL is valid
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


# Function to sanitize the URL
def sanitize_url(url):
    return urljoin(url, urlparse(url).path)


# Define and check for local file address prefixes
def check_local_file_access(url):
    local_prefixes = ['file:///', 'file://localhost', 'http://localhost', 'https://localhost']
    return any(url.startswith(prefix) for prefix in local_prefixes)


def get_response(url, headers=cfg.user_agent_header, timeout=10):
    try:
        # Restrict access to local files
        if check_local_file_access(url):
            raise ValueError('Access to local files is restricted')

        # Most basic check if the URL is valid:
        if not url.startswith('http://') and not url.startswith('https://'):
            raise ValueError('Invalid URL format')

        sanitized_url = sanitize_url(url)

        response = requests.get(sanitized_url, headers=headers, timeout=timeout)

        # Check if the response contains an HTTP error
        if response.status_code >= 400:
            return None, "Error: HTTP " + str(response.status_code) + " error"

        return response, None
    except ValueError as ve:
        # Handle invalid URL format
        return None, "Error: " + str(ve)

    except requests.exceptions.RequestException as re:
        # Handle exceptions related to the HTTP request (e.g., connection errors, timeouts, etc.)
        return None, "Error: " + str(re)


def scrape_text(url):
    """Scrape text from a webpage"""
    response, error_message = get_response(url)
    if error_message:
        return error_message

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
    response, error_message = get_response(url)
    if error_message:
        return error_message

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


def create_message(chunk, question):
    """Create a message for the user to summarize a chunk of text"""
    return {
        "role": "user",
        "content": f"\"\"\"{chunk}\"\"\" Using the above text, please answer the following question: \"{question}\" -- if the question cannot be answered using the text, please summarize the text."
    }


def summarize_text(text, question):
    """Summarize text using the LLM model"""
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
