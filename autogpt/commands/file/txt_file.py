def read_file(file_path: str) -> str:
    """Read a file and return the contents

    Args:
        filename (str): The name of the file to read

    Returns:
        str: The contents of the file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content
