
# Time Blocks Documentation

## Get Current Time

### What it is
A utility block that provides the current time in a specified format.

### What it does
Outputs the current time when triggered, allowing users to customize the time format according to their needs.

### How it works
When triggered, the block captures the current system time and formats it according to the user's specifications. If no format is specified, it uses the default format of hours:minutes:seconds.

### Inputs
- Trigger: Any input that activates the block
- Format: The desired time format (default is "HH:MM:SS")

### Outputs
- Time: The current time formatted according to specifications

### Possible use case
Creating a time-stamp for logging events or displaying the current time in a custom dashboard.

## Get Current Date

### What it is
A block that provides the current date with optional offset capabilities.

### What it does
Outputs the current date and allows users to offset it by a specified number of days, either forward or backward.

### How it works
The block takes the current date, applies any specified offset in days, and formats the result according to user preferences.

### Inputs
- Trigger: Any input that activates the block
- Offset: Number of days to adjust the date (can be positive or negative)
- Format: The desired date format (default is "YYYY-MM-DD")

### Outputs
- Date: The calculated date in the specified format

### Possible use case
Calculating due dates for projects or scheduling future events with relative dates.

## Get Current Date and Time

### What it is
A comprehensive block that combines both date and time information.

### What it does
Outputs the current date and time together in a single formatted string.

### How it works
Captures the current system date and time and combines them into a single output using the specified format.

### Inputs
- Trigger: Any input that activates the block
- Format: The desired date and time format (default is "YYYY-MM-DD HH:MM:SS")

### Outputs
- Date_Time: Combined current date and time in the specified format

### Possible use case
Creating timestamps for event logging or generating detailed temporal records.

## Countdown Timer

### What it is
A timing block that creates a delay for a specified duration.

### What it does
Waits for a specified amount of time and then outputs a message when the timer completes.

### How it works
The block takes duration inputs in various units (days, hours, minutes, seconds), combines them into a total duration, waits for that period, and then outputs the specified message.

### Inputs
- Input Message: The message to output when the timer completes (default is "timer finished")
- Seconds: Number of seconds to wait
- Minutes: Number of minutes to wait
- Hours: Number of hours to wait
- Days: Number of days to wait

### Outputs
- Output Message: The specified message after the timer completes

### Possible use case
Creating delayed actions, scheduling tasks, or implementing waiting periods in automated workflows.
