# Code Explanation

This module contains functions to execute Python file and shell commands in a Docker container.

The following are the methods in this module:
 
`execute_python_file(filename: str) -> str`: This function takes in a file name (Python file) as input and executes it in a Docker container. 

`execute_shell(command_line: str) -> str`: This function takes in a shell command as input and executes it in a Docker container. 

`execute_shell_popen(command_line) -> str`: This function takes in a shell command as input, executes it in a Docker container with Popen, and returns information about the process started.

The `we_are_running_in_a_docker_container() -> bool` function is a helper function to check if we are running in a Docker container or not.

# Example

```python
# Execute Python file
execute_python_file("test.py")

# Execute shell command
execute_shell("ls -la")

# Execute shell command with Popen
execute_shell_popen("python test.py")
```