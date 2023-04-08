# Import necessary libraries
import asyncio
from pyppeteer import launch
from config import Config
from llm_utils import create_chat_completion

# Initialize configuration
cfg = Config()

# Fetch a URL using Pyppeteer
async def fetch_url(url):
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page.goto(url)
    return page, browser

# Scrape text from a given URL
async def scrape_text(url):
    page, browser = await fetch_url(url)
    text = await page.evaluate('() => document.body.innerText')
    await browser.close()

    # Process the scraped text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return '\n'.join(chunk for chunk in chunks if chunk)

# Scrape links from a given URL
async def scrape_links(url):
    page, browser = await fetch_url(url)
    hyperlinks = await page.evaluate('''() => Array.from(document.querySelectorAll('a[href]')).map(a => ({
        text: a.innerText,
        href: a.href
    }))''')
    await browser.close()

    # Format scraped links
    return [f"{link['text']} ({link['href']})" for link in hyperlinks]

# Split text into chunks based on a maximum length
def split_text(text, max_length=8192):
    paragraphs, current_length, current_chunk = text.split("\n"), 0, []

    for paragraph in paragraphs:
        if current_length + len(paragraph) + 1 <= max_length:
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 1
        else:
            yield "\n".join(current_chunk)
            current_chunk, current_length = [paragraph], len(paragraph) + 1

    if current_chunk:
        yield "\n".join(current_chunk)

# Create a message for the chat completion
async def create_message(chunk, question):
    return {
        "role": "user",
        "content": f"\"\"\"{chunk}\"\"\" Using the above text, please answer the following question: \"{question}\" -- if the question cannot be answered using the text, please summarize the text."
    }

# Summarize text using the chat completion
async def summarize_text(text, question):
    if not text:
        return "Error: No text to summarize"

    print(f"Text length: {len(text)} characters")
    chunks, summaries = list(split_text(text)), []

    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1} / {len(chunks)}")
        message = await create_message(chunk, question)

        summary = create_chat_completion(
            model=cfg.fast_llm_model,
            messages=[message],
            max_tokens=300,
        )
        summaries.append(summary)

    print(f"Summarized {len(chunks)} chunks.")
    combined_summary = "\n".join(summaries)
    message = await create_message(combined_summary, question)

    return create_chat_completion(
        model=cfg.fast_llm_model,
        messages=[message],
        max_tokens=300,
    )


