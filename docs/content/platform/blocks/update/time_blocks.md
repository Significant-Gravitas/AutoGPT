
# Time Management Blocks Documentation

## Get Current Time

### What it is
A block that provides the current time in a specified format.

### What it does
Returns the current time according to your system clock in a customizable format.

### How it works
When triggered, the block reads your system's current time and formats it according to your specifications. If no format is specified, it uses the default format of hours:minutes:seconds (HH:MM:SS).

### Inputs
- Trigger: Any input that activates the block
- Format: Optional time format pattern (default is HH:MM:SS)

### Outputs
- Time: The current time in the specified format

### Possible use case
Creating automated time-based notifications or logging timestamps in a specific format for reports.

## Get Current Date

### What it is
A block that provides the current date with optional date offset capabilities.

### What it does
Returns the current date or a date offset by a specified number of days, formatted according to your preferences.

### How it works
Reads the system date and applies any specified offset (forward or backward in days), then formats the result according to your specifications. The default format is YYYY-MM-DD.

### Inputs
- Trigger: Any input that activates the block
- Days Offset: Number of days to adjust the date forward or backward (default is 0)
- Format: Optional date format pattern (default is YYYY-MM-DD)

### Outputs
- Date: The calculated date in the specified format

### Possible use case
Scheduling future events or calculating deadlines by offsetting from the current date.

## Get Current Date and Time

### What it is
A block that combines both date and time information into a single output.

### What it does
Provides the current date and time together in a customizable format.

### How it works
Combines the system's current date and time information into a single formatted string. The default format includes both date and time (YYYY-MM-DD HH:MM:SS).

### Inputs
- Trigger: Any input that activates the block
- Format: Optional datetime format pattern (default is YYYY-MM-DD HH:MM:SS)

### Outputs
- Date Time: The current date and time in the specified format

### Possible use case
Creating timestamps for logging events or generating datetime stamps for document creation.

## Countdown Timer

### What it is
A block that creates a customizable countdown timer with multiple time unit inputs.

### What it does
Waits for a specified duration and then outputs a message when the time is up.

### How it works
Combines all time inputs (days, hours, minutes, seconds) into a total duration, waits for that duration to pass, then delivers your specified message.

### Inputs
- Input Message: The message to display when the timer finishes (default is "timer finished")
- Seconds: Number of seconds to wait
- Minutes: Number of minutes to wait
- Hours: Number of hours to wait
- Days: Number of days to wait

### Outputs
- Output Message: The specified message that appears when the timer completes

### Possible use case
Creating delayed notifications, scheduling automated tasks, or implementing cooling-off periods in automated workflows.
