
<file_name>autogpt_platform/backend/backend/blocks/text.md</file_name>

# Text Processing Blocks Documentation

## Match Text Pattern

### What it is
A text matching tool that checks if a given text matches a specific pattern.

### What it does
Compares input text against a pattern (regex) and routes the data to different outputs based on whether there's a match or not.

### How it works
The block takes text and a pattern, checks if they match according to specified rules (case sensitivity and dot matching), and forwards the data to either a positive or negative output.

### Inputs
- Text: The text content to be checked
- Match: The pattern (regex) to look for in the text
- Data: Information to be passed through to the output
- Case Sensitive: Whether the matching should consider letter case (default: true)
- Dot All: Whether dots in the pattern match all characters (default: true)

### Outputs
- Positive: Receives the data when a match is found
- Negative: Receives the data when no match is found

### Possible use case
Sorting customer feedback messages into different categories based on specific keywords or patterns.

## Extract Text Information

### What it is
A tool that pulls out specific pieces of information from text using patterns.

### What it does
Searches through text to find and extract specific portions that match a given pattern.

### How it works
The block searches for pattern matches in the text and extracts either the whole match or specific groups within the match, with options for finding single or multiple occurrences.

### Inputs
- Text: The text to analyze
- Pattern: The pattern (regex) to use for extraction
- Group: Which part of the match to extract (default: 0)
- Case Sensitive: Whether to match exact letter case (default: true)
- Dot All: Whether dots match all characters (default: true)
- Find All: Whether to find multiple matches (default: false)

### Outputs
- Positive: The extracted text when matches are found
- Negative: The original text when no matches are found

### Possible use case
Extracting email addresses or phone numbers from a document.

## Fill Text Template

### What it is
A template-based text formatting tool.

### What it does
Creates formatted text by inserting values into a template.

### How it works
Takes a template with placeholders and replaces them with provided values to create the final text.

### Inputs
- Values: A dictionary of values to insert into the template
- Format: The template text with placeholders (using Jinja2 syntax)

### Outputs
- Output: The final formatted text

### Possible use case
Creating personalized email messages by inserting customer names and other details into a template.

## Combine Texts

### What it is
A text combination tool that joins multiple text pieces together.

### What it does
Merges multiple text inputs into a single text output, with an optional delimiter between them.

### How it works
Takes a list of text inputs and joins them together, optionally placing a specified delimiter between each piece.

### Inputs
- Input: List of text pieces to combine
- Delimiter: Text to insert between combined pieces (default: empty string)

### Outputs
- Output: The final combined text

### Possible use case
Combining separate paragraphs into a single document, or joining pieces of a message with proper spacing.

