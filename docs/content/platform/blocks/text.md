## Match Text Pattern

### What it is
A block that matches text against a specified pattern.

### What it does
This block takes input text and a pattern, then checks if the text matches the pattern. It forwards the data to either a positive or negative output based on whether a match is found.

### How it works
The block uses regular expressions to search for the specified pattern within the input text. It considers options like case sensitivity and whether the dot should match all characters.

### Inputs
| Input | Description |
|-------|-------------|
| Text | The text to be matched against the pattern |
| Match | The pattern (regular expression) to search for in the text |
| Data | Additional information to be forwarded to the output |
| Case sensitive | Option to make the match case-sensitive or not |
| Dot all | Option to make the dot character match all characters, including newlines |

### Outputs
| Output | Description |
|--------|-------------|
| Positive | The output data if a match is found |
| Negative | The output data if no match is found |

### Possible use case
Filtering customer feedback messages based on specific keywords or phrases to categorize them as positive or negative reviews.

---

## Extract Text Information

### What it is
A block that extracts specific information from text using a pattern.

### What it does
This block searches for a pattern within the input text and extracts a portion of the text based on that pattern.

### How it works
The block uses regular expressions to find the specified pattern in the text. It then extracts a particular group from the match, which can be the entire match or a specific captured group.

### Inputs
| Input | Description |
|-------|-------------|
| Text | The text from which to extract information |
| Pattern | The pattern (regular expression) used to find the desired information |
| Group | The group number to extract from the match (0 for the entire match) |
| Case sensitive | Option to make the match case-sensitive or not |
| Dot all | Option to make the dot character match all characters, including newlines |

### Outputs
| Output | Description |
|--------|-------------|
| Positive | The extracted text if a match is found |
| Negative | The original text if no match is found |

### Possible use case
Extracting phone numbers or email addresses from a large body of text, such as a customer database.

---

## Fill Text Template

### What it is
A block that fills a text template with provided values.

### What it does
This block takes a template string and a dictionary of values, then replaces placeholders in the template with the corresponding values.

### How it works
The block uses a template engine to replace placeholders in the format string with the provided values. It supports both simple placeholder replacement and more complex operations like loops.

### Inputs
| Input | Description |
|-------|-------------|
| Values | A dictionary containing the values to be inserted into the template |
| Format | The template string with placeholders for the values |

### Outputs
| Output | Description |
|--------|-------------|
| Output | The formatted text with placeholders replaced by actual values |

### Possible use case
Generating personalized email messages by filling a template with customer-specific information like name, order details, or account status.

---

## Combine Texts

### What it is
A block that combines multiple text inputs into a single output.

### What it does
This block takes a list of text inputs and joins them together, optionally using a specified delimiter.

### How it works
The block concatenates all the input texts in the order they are provided, inserting the specified delimiter (if any) between each text.

### Inputs
| Input | Description |
|-------|-------------|
| Input | A list of text strings to be combined |
| Delimiter | An optional string to be inserted between each text input (default is an empty string) |

### Outputs
| Output | Description |
|--------|-------------|
| Output | The combined text |

### Possible use case
Merging multiple parts of an address (street, city, state, zip code) into a single, formatted address string.