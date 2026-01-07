# Code Extraction

### What it is
Extracts code blocks from text and identifies their programming languages.

### What it does
Extracts code blocks from text and identifies their programming languages

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Combine Texts

### What it is
This block combines multiple input texts into a single output text.

### What it does
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

### What it does
This block triggers after a specified duration.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| input_message | Message to output after the timer finishes | Input Message | No |
| seconds | Duration in seconds | int | str | No |
| minutes | Duration in minutes | int | str | No |
| hours | Duration in hours | int | str | No |
| days | Duration in days | int | str | No |
| repeat | Number of times to repeat the timer | int | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| output_message | Message after the timer finishes | Output Message |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Extract Text Information

### What it is
This block extracts the text from the given text using the pattern (regex).

### What it does
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

### What it does
This block formats the given texts using the format template.

### How it works
<!-- MANUAL: how_it_works -->
The block uses a template engine to replace placeholders in the format string with the provided values. It supports both simple placeholder replacement and more complex operations like loops.
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| values | Values (dict) to be used in format. These values can be used by putting them in double curly braces in the format template. e.g. {{value_name}}. | Dict[str, True] | Yes |
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

### What it does
This block outputs the current date with an optional offset.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| trigger | Trigger any data to output the current date | str | Yes |
| offset | Offset in days from the current date | int | str | No |
| format_type | Format type for date output (strftime with custom format or ISO 8601) | Format Type | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| date | Current date in the specified format (default: YYYY-MM-DD) | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Get Current Date And Time

### What it is
This block outputs the current date and time.

### What it does
This block outputs the current date and time.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Get Current Time

### What it is
This block outputs the current time.

### What it does
This block outputs the current time.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Match Text Pattern

### What it is
Matches text against a regex pattern and forwards data to positive or negative output based on the match.

### What it does
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
Decodes a string containing escape sequences into actual text.

### What it does
Decodes a string containing escape sequences into actual text

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Text Replace

### What it is
This block is used to replace a text with a new text.

### What it does
This block is used to replace a text with a new text.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Text Split

### What it is
This block is used to split a text into a list of strings.

### What it does
This block is used to split a text into a list of strings.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Word Character Count

### What it is
Counts the number of words and characters in a given text.

### What it does
Counts the number of words and characters in a given text.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
