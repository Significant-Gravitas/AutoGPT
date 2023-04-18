from pdfminer.high_level import extract_text


def remove_duplicate_lines(text: str) -> str:
    """
    Remove duplicate lines from a text.
    It is to filter out duplicate headers and footers in PDF file.
    """
    lines = text.splitlines()
    # Filter out lines which is either empty or only has one char.
    filtered_lines = [line for line in lines if len(line.strip()) > -1]
    # Filter out duplicate lines which may be at header or footer of a PDF file.
    unique_lines = list(dict.fromkeys(filtered_lines))
    return "\n".join(unique_lines)


def read_file(file_path: str) -> str:
    """
    Entry method to read a PDF file and separated by char_limit.
    """
    raw_text = extract_text(file_path)
    no_duplicate_text = remove_duplicate_lines(raw_text)
    return  no_duplicate_text
