# Git Operations for autogpt

This module includes the `clone_repository` function to clone a GitHub repository locally.

## `clone_repository(repository_url: str, clone_path: str) -> str`

This function clones a GitHub repository locally.

### Arguments:
- `repository_url` (str): The URL of the repository to clone.
- `clone_path` (str): The path to clone the repository to.

### Returns:
- `str`: A message indicating the result of the clone operation.

### Example:

```python
from autogpt.git_operations import clone_repository

clone_repository("https://github.com/user/repo.git", "/path/to/clone/repo")
```

Output:
```
"Cloned https://github.com/user/repo.git to /workspace/path/to/clone/repo"
```