## Get Current Time

### What it is
A block that provides the current time.

### What it does
This block outputs the current time in hours, minutes, and seconds.

### How it works
When triggered, the block retrieves the current system time and formats it as a string in the HH:MM:SS format.

### Inputs
| Input | Description |
|-------|-------------|
| trigger | A string input that activates the block. The content of this input doesn't affect the output. |

### Outputs
| Output | Description |
|--------|-------------|
| time | A string representing the current time in the format HH:MM:SS (e.g., "14:30:45"). |

### Possible use case
This block could be used in a chatbot that needs to provide the current time to users when asked.

---

## Get Current Date

### What it is
A block that provides the current date, with an optional offset.

### What it does
This block outputs the current date or a date offset from the current date by a specified number of days.

### How it works
When triggered, the block retrieves the current system date. If an offset is provided, it calculates a new date by subtracting the offset number of days from the current date. The resulting date is then formatted as a string in the YYYY-MM-DD format.

### Inputs
| Input | Description |
|-------|-------------|
| trigger | A string input that activates the block. The content of this input doesn't affect the output. |
| offset | An integer or string representing the number of days to subtract from the current date. If not provided or invalid, it defaults to 0. |

### Outputs
| Output | Description |
|--------|-------------|
| date | A string representing the date in the format YYYY-MM-DD (e.g., "2023-05-15"). |

### Possible use case
This block could be used in a scheduling application to calculate and display dates for upcoming events or deadlines.

---

## Get Current Date and Time

### What it is
A block that provides both the current date and time.

### What it does
This block outputs the current date and time combined into a single string.

### How it works
When triggered, the block retrieves the current system date and time, then formats them together as a string in the YYYY-MM-DD HH:MM:SS format.

### Inputs
| Input | Description |
|-------|-------------|
| trigger | A string input that activates the block. The content of this input doesn't affect the output. |

### Outputs
| Output | Description |
|--------|-------------|
| date_time | A string representing the current date and time in the format YYYY-MM-DD HH:MM:SS (e.g., "2023-05-15 14:30:45"). |

### Possible use case
This block could be used in a logging system to timestamp events with both date and time information.

---

## Countdown Timer

### What it is
A block that acts as a countdown timer, triggering after a specified duration.

### What it does
This block waits for a specified amount of time and then outputs a message.

### How it works
The block takes input for the duration in days, hours, minutes, and seconds. It calculates the total wait time in seconds, pauses execution for that duration, and then outputs the specified message.

### Inputs
| Input | Description | Default |
|-------|-------------|---------|
| input_message | The message to be output when the timer finishes. | "timer finished" |
| seconds | The number of seconds to wait. | 0 |
| minutes | The number of minutes to wait. | 0 |
| hours | The number of hours to wait. | 0 |
| days | The number of days to wait. | 0 |

### Outputs
| Output | Description |
|--------|-------------|
| output_message | The message specified in the input_message, output after the timer completes. |

### Possible use case
This block could be used in a reminder application to trigger notifications after a set amount of time, or in a cooking app to notify users when a recipe step is complete.