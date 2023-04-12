# Auto GPT File System Manager
This is a collection of helper functions to manage files in the Auto GPT framework. 

## Functions:

### `safe_join(base, *paths)`
Join one or more path components intelligently.

#### Args:
- `base` (`str`): Base path for the file to be created/read.
- `*paths` (`str`, optional): Any additional paths to be joined with the base path.

#### Returns:
- `new_path` (`str`): The joined paths.

#### Raises:
- `ValueError`: if the resulting path would access a file outside of the `working_directory`.

### `read_file(filename)`
Read a file and return the contents.

#### Args:
- `filename` (`str`): Name of the file to be read.

#### Returns:
- `content` (`str`): The contents of the file.

#### Raises:
- `Exception`: If file I/O fails.

### `write_to_file(filename, text)`
Write text to a file.

#### Args:
- `filename` (`str`): Name of the file to be written.
- `text` (`str`): Text to be written to the file.

#### Returns:
- `str`: A success/fail message. 

#### Raises:
- `Exception`: If file I/O fails.

### `append_to_file(filename, text)`
Append text to a file.

#### Args:
- `filename` (`str`): Name of the file to be appended.
- `text` (`str`): Text to be appended to the file.

#### Returns:
- `str`: A success/fail message. 

#### Raises:
- `Exception`: If file I/O fails.

### `delete_file(filename)`
Delete a file.

#### Args:
- `filename` (`str`): Name of the file to be deleted.

#### Returns:
- `str`: A success/fail message. 

#### Raises:
- `Exception`: If file I/O fails.

### `search_files(directory)`
Return list of files in the specified directory.

#### Args:
- `directory` (`str`): Path of the directory to be searched. Defaults to the `working_directory` if not provided.

#### Returns:
- `found_files` (`list`): A list of relative file paths.

#### Raises:
- `Exception`: If file I/O fails.