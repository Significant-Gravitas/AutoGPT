# `execute_python_file` Function

This function executes a Python file inside a Docker container and returns the output.

## Arguments
- `file` (string): Name of the Python file to execute.

## Returns
- (string): Output of running the Python file.

## Example Usage
```python
output = execute_python_file("example.py")
print(output)
```

In the example above, the `execute_python_file` function would execute the `example.py` script inside a Docker container and return the output. The output is then printed to the console. Please ensure docker is installed on your system to avoid errors.