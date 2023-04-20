## Function `get_datetime()`

Returns the current date and time as a string.

### Arguments and Return value:

- Argument: None
- Return value: 
    - `str`: The current date and time

### Example usage:

```
from datetime import datetime

def get_datetime() -> str:
    """Return the current date and time

    Returns:
        str: The current date and time
    """
    return "Current date and time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

current_datetime = get_datetime()

# Output
print(current_datetime)
# Current date and time: 2021-09-15 16:30:00
```