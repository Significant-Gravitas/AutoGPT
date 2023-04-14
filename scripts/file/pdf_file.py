import os
import re
from pdfminer.high_level import extract_text

def filter_text(text):
    """
    T reduce the PDF file size for most scientific articles, remove all text
    following by 'REFERENCE' section (which should be in a whole line) and starts with [\d] at each lines beginning.
    """
    text = re.sub(r"(?i)^\s*REFERENCES\s*$[\s\S]*", "", text, flags=re.MULTILINE)  # Remove text after "REFERENCES" (case-insensitive)
    paragraphs = re.split(r'\n\s*\n', text)
    filtered_paragraphs = [p for p in paragraphs if not re.match(r"\s*\[\d+\]", p.strip())]
    return "\n\n".join(filtered_paragraphs)


def remove_duplicate_lines(text):
    lines = text.splitlines()
    # Filter out lines which is either empty or only has one char.
    filtered_lines = [line for line in lines if len(line.strip()) > -1]
    # Filter out duplicate lines which may be at header or footer of a PDF file.
    unique_lines = list(dict.fromkeys(filtered_lines))
    return "\n".join(unique_lines)


def split_text(text, char_limit):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk + paragraph) <= char_limit:
            current_chunk += "\n\n" + paragraph
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def read_file(file_path, char_limit=4000):
    """
    Entry method to read a PDF file and separated by char_limit.
    """
    raw_text = extract_text(file_path)
    filtered_text = filter_text(raw_text)
    no_duplicate_text = remove_duplicate_lines(filtered_text)
    chunks = split_text(no_duplicate_text, char_limit)

    content = f"\n===CHUNK===\n".join(chunks)
    return content
