import re

from config import Config

cfg = Config()

def create_message(chunk):
    """Make key points or outline the text content"""
    return {
        "role": "user",
        "content": f"\"\"\"{chunk}\"\"\" List all the questions I can ask about the text and their condensed answers."
    }


def split_text(text, char_limit=4000):
    # Regex pattern for splitting at two line breaks, one line break, or a period
    pattern = r"(\n\n|\n|\.)"
    words = re.split(pattern, text)

    chunks = []
    chunk = ""
    chunk_size = 0

    for word in words:
        word_size = len(word)
        if chunk_size + word_size < char_limit:
            chunk += word
            chunk_size += word_size
        else:
            chunks.append(chunk)
            chunk = word
            chunk_size = word_size

    if chunk:
        chunks.append(chunk)

    return chunks
