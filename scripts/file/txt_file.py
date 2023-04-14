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


def read_file(filepath, char_limit=4000):
    """Read a file and return the contents"""
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            content = f.read()
            chunks = split_text(content, char_limit)
            content = f"\nCHUNK\n".join(chunks)
            return content
    except Exception as e:
        return "Error: " + str(e)
