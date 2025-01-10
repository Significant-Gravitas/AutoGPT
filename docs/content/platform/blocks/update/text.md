
## Get Current Time

### What it is
A block that provides the current time in a customizable format.

### What it does
Outputs the current time when triggered, allowing users to specify their preferred time format.

### How it works
When triggered, the block captures the current system time and formats it according to the user's specifications. If no format is specified, it uses the default format of hours:minutes:seconds.

### Inputs
- Trigger: Any input that activates the block to output the current time
- Format: Optional formatting pattern for the time output (default: HH:MM:SS)

### Outputs
- Time: The current time formatted according to the specified pattern

### Possible use case
Creating a digital clock display for a dashboard or logging timestamps in a specific format for event tracking.

## Get Current Date

### What it is
A block that provides the current date with optional day offsetting capabilities.

### What it does
Outputs the current date and allows users to offset it by a specified number of days (forward or backward) with customizable formatting.

### How it works
The block takes the current date, applies any specified day offset, and formats the result according to the user's preferences. If no format is specified, it uses the standard YYYY-MM-DD format.

### Inputs
- Trigger: Any input that activates the block to output the date
- Days Offset: Number of days to adjust the date forward or backward (default: 0)
- Format: Optional formatting pattern for the date output (default: YYYY-MM-DD)

### Outputs
- Date: The calculated date in the specified format

### Possible use case
Scheduling applications, deadline calculations, or generating dates for report generation with consistent formatting.

## Get Current Date and Time

### What it is
A block that combines both date and time information into a single output.

### What it does
Provides a complete timestamp containing both the current date and time in a customizable format.

### How it works
Captures the current system date and time and combines them into a single formatted string according to the user's specifications.

### Inputs
- Trigger: Any input that activates the block to output the timestamp
- Format: Optional formatting pattern for the combined date and time (default: YYYY-MM-DD HH:MM:SS)

### Outputs
- Date Time: The current timestamp in the specified format

### Possible use case
Creating log entries, generating timestamps for database records, or displaying a complete datetime stamp on user interfaces.

## Countdown Timer

### What it is
A block that creates a customizable countdown timer with multiple time unit inputs.

### What it does
Waits for a specified duration and then outputs a message when the time has elapsed.

### How it works
Combines the input time units (days, hours, minutes, and seconds) into a total duration, waits for that duration to pass, and then outputs the specified message.

### Inputs
- Input Message: The message to display when the timer completes (default: "timer finished")
- Seconds: Number of seconds to wait (default: 0)
- Minutes: Number of minutes to wait (default: 0)
- Hours: Number of hours to wait (default: 0)
- Days: Number of days to wait (default: 0)

### Outputs
- Output Message: The specified message that is output when the timer completes

### Possible use case
Creating delayed notifications, scheduling automated tasks, or implementing cooldown periods in applications.

