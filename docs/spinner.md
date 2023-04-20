# Spinner

A simple module to display a spinning animation in the terminal while a program is running.

## How to Use

To use this module, you can import the `Spinner` class and initialize an instance. You can specify a message and a delay between each update. The delay value determines how fast the spinner rotates.

Here is an example:

```python
from spinner import Spinner
import time

with Spinner("Loading..."):
    # do something that takes time
    time.sleep(5)
```

You can also update the spinner message in case you want to keep the user informed about the progress:

```python
with Spinner("Loading...") as spinner:
    spinner.update_message("Processing data...")
    # do some processing...
    spinner.update_message("Finished!")
```

## API

### `Spinner` class
A class to display a spinner animation in the terminal.

#### `__init__(self, message: str = "Loading...", delay: float = 0.1) -> None`
Initialize the spinner class
* `message` (str): The message to display.
* `delay` (float): The delay between each spinner update.

#### `__enter__(self)`
Start the spinning animation.

#### `__exit__(self, exc_type, exc_value, exc_traceback) -> None`
Stop the spinning animation.
* `exc_type` (Exception): The exception type.
* `exc_value` (Exception): The exception value.
* `exc_traceback` (Exception): The exception traceback.

#### `update_message(self, new_message: str, delay: float = 0.1) -> None`
Update the spinner message
* `new_message` (str): New message to display
* `delay`: Delay in seconds before updating the message. The default value is 0.1 seconds.