## File Operations for AutoGPT

This module contains functions and commands for performing file operations in the AutoGPT workspace.

### `check_duplicate_operation(operation: str, filename: str) -> bool`

Check if the operation has already been performed on the given file.

##### Args:
- `operation` (str): The operation to check for.
- `filename` (str): The name of the file to check for.

##### Returns:
- `bool`: True if the operation has already been performed on the file.

### `log_operation(operation: str, filename: str) -> None`

Log the file operation to the `file_logger.txt`.

##### Args:
- `operation` (str): The operation to log.
- `filename` (str): The name of the file the operation was performed on.

##### Returns:
- `None`

### `split_file(content: str, max_length: int = 4000, overlap: int = 0) -> Generator[str, None, None]`

Split text into chunks of a specified maximum length with a specified overlap between chunks.

##### Args:
- `content` (str): The input text to be split into chunks.
- `max_length` (int, default=4000): The maximum length of each chunk, about 1k token.
- `overlap` (int, default=0): The number of overlapping characters between chunks.

##### Returns:
- `Generator`: A generator yielding chunks of text.

### `read_file(filename: str) -> str`

Read a file and return the contents.

##### Args:
- `filename` (str): The name of the file to read.

##### Returns:
- `str`: The contents of the file.

### `ingest_file(filename: str, memory, max_length: int = 4000, overlap: int = 200) -> None`

Ingest a file by reading its content, splitting it into chunks with a specified maximum length and overlap, and adding the chunks to the memory storage.

##### Args:
- `filename` (str): The name of the file to ingest.
- `memory` (object): An object with an `add()` method to store the chunks in memory.
- `max_length` (int, default=4000): The maximum length of each chunk.
- `overlap` (int, default=200): The number of overlapping characters between chunks.

##### Returns:
- `None`

### `write_to_file(filename: str, text: str) -> str`

Write text to a file.

##### Args:
- `filename` (str): The name of the file to write to.
- `text` (str): The text to write to the file.

##### Returns:
- `str`: A message indicating success or failure.

### `append_to_file(filename: str, text: str, shouldLog: bool = True) -> str`

Append text to a file.

##### Args:
- `filename` (str): The name of the file to append to.
- `text` (str): The text to append to the file.
- `shouldLog` (bool, default=True): A flag to indicate if the file operation should be logged.

##### Returns:
- `str`: A message indicating success or failure.

### `delete_file(filename: str) -> str`

Delete a file.

##### Args:
- `filename` (str): The name of the file to delete.

##### Returns:
- `str`: A message indicating success or failure.

### `search_files(directory: str) -> list[str]`

Search for files in a directory.

##### Args:
- `directory` (str): The directory to search in.

##### Returns:
- `list[str]`: A list of files found in the directory.

### `download_file(url: str, filename: str)`

Downloads a file.

##### Args:
- `url` (str): URL of the file to download.
- `filename` (str): Filename to save the file as.

##### Returns:
- `str`: A message indicating success or failure.