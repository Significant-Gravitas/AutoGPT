

# Text Processing Blocks Documentation

## Match Text Pattern

### What it is
A text pattern matching tool that checks if a given text matches a specified pattern.

### What it does
Examines input text against a pattern and routes the data to different outputs based on whether a match is found or not.

### How it works
The block takes text and a pattern, checks if they match according to specified rules (like case sensitivity), and directs the data to either a positive or negative output channel.

### Inputs
- Text: The text content to be analyzed
- Match: The pattern to look for in the text
- Data: Information to be passed through to the output
- Case Sensitive: Whether uppercase and lowercase letters should be treated as different
- Dot All: Whether the dot character should match any character including newlines

### Outputs
- Positive: Forwards the data when a pattern match is found
- Negative: Forwards the data when no pattern match is found

### Possible use case
Sorting customer feedback messages into different categories based on specific keywords or patterns.

## Extract Text Information

### What it is
A text extraction tool that pulls out specific information from text based on patterns.

### What it does
Searches through text to find and extract specific portions that match a given pattern.

### How it works
The block searches for pattern matches in the text and extracts either the whole match or specific groups within the match, with options for finding single or multiple occurrences.

### Inputs
- Text: The source text to extract information from
- Pattern: The pattern that defines what to extract
- Group: Which part of the match to extract
- Case Sensitive: Whether to consider letter case during matching
- Dot All: Whether the dot character matches any character including newlines
- Find All: Whether to find one or all matches

### Outputs
- Positive: The extracted text when matches are found
- Negative: The original text when no matches are found

### Possible use case
Extracting phone numbers or email addresses from a document.

## Fill Text Template

### What it is
A template-based text formatter that creates customized text using provided values.

### What it does
Fills in a text template with specific values to create formatted text output.

### How it works
The block takes a template with placeholders and replaces them with corresponding values from a provided dictionary.

### Inputs
- Values: A dictionary of values to insert into the template
- Format: The template text with placeholders for the values

### Outputs
- Output: The final formatted text with all placeholders replaced with values

### Possible use case
Creating personalized email messages by filling in customer names and specific details in a template.

## Combine Texts

### What it is
A text combination tool that joins multiple text pieces into a single text.

### What it does
Merges multiple separate text inputs into one unified text output, with an optional delimiter between them.

### How it works
The block takes a list of text pieces and joins them together, optionally placing a specified delimiter between each piece.

### Inputs
- Input: A list of text pieces to combine
- Delimiter: The text to insert between combined pieces (optional)

### Outputs
- Output: The final combined text

### Possible use case
Combining multiple paragraphs into a single document, or joining separate pieces of information with commas or spaces between them.

