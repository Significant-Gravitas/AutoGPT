# Logger module for Auto-GPT

This module provides a `Logger` class that handles logging in different colors and outputs logs in console, activity.log, and errors.log. It also creates a log directory if it doesn't exist.

The `Logger` class has the following methods for logging:

- `typewriter_log`: Logs a message with a title in bold and a specified color. Optionally, speaks the message in the speak mode.
- `debug`: Logs a debug message with an optional title and title color.
- `warn`: Logs a warning message with an optional title and title color.
- `error`: Logs an error message with a required title and an optional message.
- `set_level`: Sets the logging level of the logger.
- `double_check`: Logs a warning with a title to remind users to double-check the configuration.

This module also provides two custom `logging.Handler` subclasses:

- `TypingConsoleHandler`: Outputs logs to console using simulated typing.
- `AutoGptFormatter`: Allows the use of custom placeholders `title_color` and `message_no_color` in the log formatting.

Additionally, this module provides a helper function `print_assistant_thoughts` that prints the assistant's thoughts to the console and speaks them in the speak mode.