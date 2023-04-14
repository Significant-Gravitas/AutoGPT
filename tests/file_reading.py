import os
import re
from pdfminer.high_level import extract_text

"""
author wangqi
"""

def filter_text(text):
    text = re.sub(r"(?i)^\s*REFERENCES\s*$[\s\S]*", "", text, flags=re.MULTILINE)  # Remove text after "REFERENCES" (case-insensitive)
    paragraphs = re.split(r'\n\s*\n', text)
    filtered_paragraphs = [p for p in paragraphs if not re.match(r"\s*\[\d+\]", p.strip())]
    return "\n\n".join(filtered_paragraphs)

def remove_duplicate_lines(text):
    lines = text.splitlines()
    unique_lines = list(dict.fromkeys(lines))
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

def print_chunks(file_path, char_limit):
    raw_text = extract_text(file_path)
    filtered_text = filter_text(raw_text)
    no_duplicate_text = remove_duplicate_lines(filtered_text)
    chunks = split_text(no_duplicate_text, char_limit)

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ---")
        print(chunk)
        print("\n" + "=" * 40 + "\n")

    file_path = os.path.join("auto_gpt_workspace", "agent.txt")
    with open(file_path, 'w') as out_file:
        for chunk in chunks:
            out_file.write(chunk)

if __name__ == "__main__":
    file_path = os.path.join("auto_gpt_workspace", "agent.pdf")
    char_limit = 4000
    print_chunks(file_path, char_limit)
