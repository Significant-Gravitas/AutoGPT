# `load_prompt()` function

This function loads the prompt from a file located at the `data/prompt.txt` path relative to the current file.

```python
def load_prompt() -> str:
    """
    Load the prompt from data/prompt.txt
    
    Returns:
        str: The content of the prompt file.
    """
```

## Parameters

This function takes no parameters.


## Returns

The function returns a string containing the content of the prompt file.


## Example usage

```python
prompt = load_prompt()
print(prompt)
``` 

Output:
```
Hello, how can I help you today?
``` 

If the prompt file does not exist at the expected location, an empty string is returned and an error message is printed to the console.

Note: this code depends on the `os` and `Path` modules from Python's standard library, so make sure they are imported before calling this function.