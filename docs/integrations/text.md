# Text
<!-- MANUAL: file_description -->
Blocks for text processing including formatting, extraction, transformation, splitting, combining, and template rendering.
<!-- END MANUAL -->

## Code Extraction

### What it is
Extracts code blocks from text and identifies their programming languages

### How it works
<!-- MANUAL: how_it_works -->
This block parses text content (typically from AI responses) and extracts code blocks enclosed in markdown-style triple backticks. It identifies the programming language from the code fence annotation (e.g., ```python) and routes each extracted code block to the appropriate language-specific output.

The block supports 16 programming languages including Python, JavaScript, HTML, CSS, SQL, and more. Any text that remains after extracting all code blocks is output as "remaining_text", allowing you to process both the code and surrounding context separately.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| text | Text containing code blocks to extract (e.g., AI response) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| html | Extracted HTML code | str |
| css | Extracted CSS code | str |
| javascript | Extracted JavaScript code | str |
| python | Extracted Python code | str |
| sql | Extracted SQL code | str |
| java | Extracted Java code | str |
| cpp | Extracted C++ code | str |
| csharp | Extracted C# code | str |
| json_code | Extracted JSON code | str |
| bash | Extracted Bash code | str |
| php | Extracted PHP code | str |
| ruby | Extracted Ruby code | str |
| yaml | Extracted YAML code | str |
| markdown | Extracted Markdown code | str |
| typescript | Extracted TypeScript code | str |
| xml | Extracted XML code | str |
| remaining_text | Remaining text after code extraction | str |

### Possible use case
<!-- MANUAL: use_case -->
**AI Code Generation Pipeline**: When an AI model generates a response containing multiple code blocks (HTML, CSS, JavaScript), use this block to separate each language into individual files for a complete web component.

**Code Review Automation**: Extract Python code from documentation or chat logs to run automated linting, testing, or security scanning on the code portions only.

**Technical Documentation Processing**: Parse developer tutorials or README files to extract executable code samples while preserving the explanatory text for different processing paths.
<!-- END MANUAL -->

---

## Combine Texts

### What it is
This block combines multiple input texts into a single output text.

### How it works
<!-- MANUAL: how_it_works -->
The block concatenates all the input texts in the order they are provided, inserting the specified delimiter (if any) between each text.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| input | text input to combine | List[str] | Yes |
| delimiter | Delimiter to combine texts | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| output | Combined text | str |

### Possible use case
<!-- MANUAL: use_case -->
Merging multiple parts of an address (street, city, state, zip code) into a single, formatted address string.
<!-- END MANUAL -->

---

## Countdown Timer

### What it is
This block triggers after a specified duration.

### How it works
<!-- MANUAL: how_it_works -->
The Countdown Timer block pauses workflow execution for a specified duration before continuing. You can set the delay using any combination of seconds, minutes, hours, and days. When the timer completes, it outputs your specified message (or "timer finished" by default).

The block supports a repeat parameter, allowing the timer to fire multiple times in sequence—useful for creating periodic triggers within your workflow. The timer uses async sleep, so it doesn't block other concurrent operations in the system.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| input_message | Message to output after the timer finishes | Input Message | No |
| seconds | Duration in seconds | int \| str | No |
| minutes | Duration in minutes | int \| str | No |
| hours | Duration in hours | int \| str | No |
| days | Duration in days | int \| str | No |
| repeat | Number of times to repeat the timer | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| output_message | Message after the timer finishes | Output Message |

### Possible use case
<!-- MANUAL: use_case -->
**Rate Limiting**: Add a delay between API calls to respect rate limits when processing large batches of data through external services.

**Scheduled Notifications**: Create a workflow that waits a specific time after an event (like a form submission) before sending a follow-up email or notification.

**Polling Workflows**: Use the repeat feature to periodically check for updates, such as monitoring a file location or checking an API endpoint every few minutes.
<!-- END MANUAL -->

---

## Extract Text Information

### What it is
This block extracts the text from the given text using the pattern (regex).

### How it works
<!-- MANUAL: how_it_works -->
The block uses regular expressions to find the specified pattern in the text. It then extracts a particular group from the match, which can be the entire match or a specific captured group.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| text | Text to parse | Text | Yes |
| pattern | Pattern (Regex) to parse | str | Yes |
| group | Group number to extract | int | No |
| case_sensitive | Case sensitive match | bool | No |
| dot_all | Dot matches all | bool | No |
| find_all | Find all matches | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| positive | Extracted text | str |
| negative | Original text | str |
| matched_results | List of matched results | List[str] |
| matched_count | Number of matched results | int |

### Possible use case
<!-- MANUAL: use_case -->
Extracting phone numbers or email addresses from a large body of text, such as a customer database.
<!-- END MANUAL -->

---

## Fill Text Template

### What it is
This block formats the given texts using the format template.

### How it works
<!-- MANUAL: how_it_works -->
The block uses a template engine to replace placeholders in the format string with the provided values. It supports both simple placeholder replacement and more complex operations like loops.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| values | Values (dict) to be used in format. These values can be used by putting them in double curly braces in the format template. e.g. {{value_name}}. | Dict[str, Any] | Yes |
| format | Template to format the text using `values`. Use Jinja2 syntax. | str | Yes |
| escape_html | Whether to escape special characters in the inserted values to be HTML-safe. Enable for HTML output, disable for plain text. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| output | Formatted text | str |

### Possible use case
<!-- MANUAL: use_case -->
Generating personalized email messages by filling a template with customer-specific information like name, order details, or account status.
<!-- END MANUAL -->

---

## Get Current Date

### What it is
This block outputs the current date with an optional offset.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves the current date from the system clock and formats it according to your preferences. You can specify an offset in days to get past or future dates (negative values for past dates, positive for future).

The block supports two format types: strftime (customizable format strings like "%Y-%m-%d" or "%B %d, %Y") and ISO 8601 (standard format YYYY-MM-DD). You can also configure the timezone—either specify a specific timezone or use the user's profile timezone setting.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| trigger | Trigger any data to output the current date | str | Yes |
| offset | Offset in days from the current date | int \| str | No |
| format_type | Format type for date output (strftime with custom format or ISO 8601) | Format Type | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| date | Current date in the specified format (default: YYYY-MM-DD) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Daily Report Generation**: Use the current date as a filename suffix or report header when generating automated daily summaries or exports.

**Deadline Tracking**: Calculate dates relative to today using the offset feature—for example, find the date 30 days from now for payment due dates or project milestones.

**Date-Based Filtering**: Get today's date to filter records, events, or tasks that are relevant to the current day in your workflow.
<!-- END MANUAL -->

---

## Get Current Date And Time

### What it is
This block outputs the current date and time.

### How it works
<!-- MANUAL: how_it_works -->
This block outputs the current date and time from the system clock, formatted according to your specifications. It supports both strftime custom formats (like "%Y-%m-%d %H:%M:%S") and ISO 8601/RFC 3339 format for maximum compatibility with APIs and databases.

You can configure the timezone to use either a specific timezone (e.g., "America/New_York") or the user's profile timezone. The ISO 8601 format option includes an optional microseconds precision setting for applications requiring high-resolution timestamps.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| trigger | Trigger any data to output the current date and time | str | Yes |
| format_type | Format type for date and time output (strftime with custom format or ISO 8601/RFC 3339) | Format Type | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| date_time | Current date and time in the specified format (default: YYYY-MM-DD HH:MM:SS) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Audit Logging**: Add precise timestamps to records when logging user actions, system events, or data changes in your workflow.

**API Requests**: Generate ISO 8601 formatted timestamps required by many REST APIs for request authentication or data submission.

**Scheduling Logic**: Compare the current date/time against scheduled events to trigger time-sensitive automations like sending reminders or processing batch jobs.
<!-- END MANUAL -->

---

## Get Current Time

### What it is
This block outputs the current time.

### How it works
<!-- MANUAL: how_it_works -->
This block outputs just the current time (without date) from the system clock. It supports strftime custom formats (like "%H:%M:%S" for 24-hour time or "%I:%M %p" for 12-hour time with AM/PM) and ISO 8601 time format.

The timezone can be configured to a specific timezone or to use the user's profile timezone setting. When using ISO 8601 format, the output includes the timezone offset and an optional microseconds component for precision timing needs.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| trigger | Trigger any data to output the current time | str | Yes |
| format_type | Format type for time output (strftime with custom format or ISO 8601) | Format Type | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| time | Current time in the specified format (default: %H:%M:%S) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Business Hours Check**: Get the current time to determine if a request falls within business hours and route it accordingly (live support vs. after-hours message).

**Time-Based Greetings**: Generate personalized greetings ("Good morning", "Good afternoon") based on the current time of day.

**Shift Scheduling**: Determine which team or process should handle a task based on the current time and configured shift schedules.
<!-- END MANUAL -->

---

## Match Text Pattern

### What it is
Matches text against a regex pattern and forwards data to positive or negative output based on the match.

### How it works
<!-- MANUAL: how_it_works -->
The block uses regular expressions to search for the specified pattern within the input text. It considers options like case sensitivity and whether the dot should match all characters.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| text | Text to match | Text | Yes |
| match | Pattern (Regex) to match | str | Yes |
| data | Data to be forwarded to output | Data | Yes |
| case_sensitive | Case sensitive match | bool | No |
| dot_all | Dot matches all | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| positive | Output data if match is found | Positive |
| negative | Output data if match is not found | Negative |

### Possible use case
<!-- MANUAL: use_case -->
Filtering customer feedback messages based on specific keywords or phrases to categorize them as positive or negative reviews.
<!-- END MANUAL -->

---

## Text Decoder

### What it is
Decodes a string containing escape sequences into actual text

### How it works
<!-- MANUAL: how_it_works -->
This block processes a text string and converts escape sequences (like \n, \t, \\, \uXXXX) into their actual character representations. For example, the literal text "Hello\nWorld" becomes "Hello" followed by an actual newline, then "World".

This is useful when working with data from APIs or files where escape sequences are stored as literal text rather than being interpreted as special characters. The block handles standard escape sequences including newlines, tabs, Unicode characters, and more.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| text | A string containing escaped characters to be decoded | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| decoded_text | The decoded text with escape sequences processed | str |

### Possible use case
<!-- MANUAL: use_case -->
**JSON Data Processing**: When parsing JSON strings that contain escaped characters (like "\n" for newlines in a message field), decode them to display properly formatted text to users.

**Log File Processing**: Process log entries where special characters are escaped, converting them to their actual representations for proper parsing or display.

**API Response Handling**: Decode text from APIs that return escaped content, ensuring special characters like tabs and newlines render correctly in your output.
<!-- END MANUAL -->

---

## Text Replace

### What it is
This block is used to replace a text with a new text.

### How it works
<!-- MANUAL: how_it_works -->
This block performs a simple find-and-replace operation on text. It searches for all occurrences of the "old" string within your input text and replaces each one with the "new" string. The replacement is case-sensitive and matches exact strings.

Unlike regex-based replacements, this block performs literal string matching, making it straightforward and predictable for common text substitution tasks. All instances of the target string are replaced in a single operation.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| text | The text to replace. | str | Yes |
| old | The old text to replace. | str | Yes |
| new | The new text to replace with. | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| output | The text with the replaced text. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Data Sanitization**: Replace sensitive information like placeholder tokens with actual values, or redact personal information before displaying or storing text.

**Template Customization**: Swap out placeholder text in templates (like "[COMPANY_NAME]" or "{{user}}") with actual values before sending emails or generating documents.

**URL Manipulation**: Modify URLs by replacing domain names, query parameters, or path segments to redirect requests or update links dynamically.
<!-- END MANUAL -->

---

## Text Split

### What it is
This block is used to split a text into a list of strings.

### How it works
<!-- MANUAL: how_it_works -->
This block takes a text string and divides it into a list of substrings based on a specified delimiter. For example, splitting "apple,banana,cherry" by comma results in ["apple", "banana", "cherry"].

By default, the block also strips whitespace from each resulting substring (controlled by the "strip" option). If your input text is empty, the block returns an empty list. This is useful for parsing structured text data like CSV values, tags, or any delimited content.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| text | The text to split. | str | Yes |
| delimiter | The delimiter to split the text by. | str | Yes |
| strip | Whether to strip the text. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| texts | The text split into a list of strings. | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Tag Processing**: Split a comma-separated list of tags or keywords into individual items for processing, filtering, or database storage.

**Line-by-Line Processing**: Split a multi-line text file by newline characters to process each line independently in your workflow.

**Parsing User Input**: Break apart user-submitted lists (like email addresses separated by semicolons) into individual items for validation and processing.
<!-- END MANUAL -->

---

## Word Character Count

### What it is
Counts the number of words and characters in a given text.

### How it works
<!-- MANUAL: how_it_works -->
This block analyzes input text and returns two metrics: the total number of words and the total number of characters. Words are counted by splitting the text on whitespace, so "Hello World" counts as 2 words. Characters include all characters in the text, including spaces and punctuation.

This provides a quick way to measure text length for validation, summarization checks, or content analysis without needing custom logic.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| text | Input text to count words and characters | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the counting operation failed | str |
| word_count | Number of words in the input text | int |
| character_count | Number of characters in the input text | int |

### Possible use case
<!-- MANUAL: use_case -->
**Content Validation**: Check if user-submitted text (like reviews or comments) meets minimum or maximum length requirements before accepting it.

**AI Prompt Optimization**: Measure prompt length to ensure it fits within token limits before sending to language models, potentially triggering summarization if too long.

**Social Media Preparation**: Verify that text content fits within platform character limits (like Twitter's 280 characters) before attempting to post.
<!-- END MANUAL -->

---
