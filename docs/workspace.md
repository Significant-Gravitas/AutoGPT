### Module AutoGPT Workspace Utils

This module contains utilities for handling files and directories within the AutoGPT workspace.

#### Importing

To use the functions defined in this module, import it in your Python file like this:

```python
from autogpt.workspace_utils import path_in_workspace, safe_path_join
```

#### Functions

##### `path_in_workspace(relative_path: str | Path) -> Path`

Get full path for item in workspace

###### Parameters:
- `relative_path` (str|Path): Path to translate into the workspace

###### Returns:
- `Path`: Absolute path for the given path in the workspace

###### Example
```python
>>> from autogpt.workspace_utils import path_in_workspace
>>> path_in_workspace("data/some_file.txt")
PosixPath('/path/to/workspace/data/some_file.txt')
```

##### `safe_path_join(base: Path, *paths: str|Path) -> Path`

Join one or more path components, asserting the resulting path is within the workspace.

###### Parameters:
- `base` (Path): The base path
- `*paths` (str|Path): The paths to join to the base path

###### Returns:
- `Path`: The joined path

###### Example
```python
>>> from autogpt.workspace_utils import safe_path_join
>>> WORKSPACE_PATH = Path("/path/to/workspace")
>>> safe_path_join(WORKSPACE_PATH, "data", "some_file.txt")
PosixPath('/path/to/workspace/data/some_file.txt')
```