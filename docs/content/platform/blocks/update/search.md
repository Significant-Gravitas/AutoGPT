

## Match Text Pattern

### What it is
A tool that checks if a specific pattern exists within a text and routes data accordingly.

### What it does
Searches through text to find matches for a specific pattern and directs the output to different paths depending on whether a match is found or not.

### How it works
The block takes input text and a pattern to search for, then examines the text to see if the pattern exists. If it finds a match, it sends the data through the "positive" path; if not, it sends it through the "negative" path.

### Inputs
- Text: The text content to be searched
- Match: The pattern (in regular expression format) to search for
- Data: Information that will be passed to the output
- Case Sensitive: Whether the search should consider upper/lowercase differences (default: yes)
- Dot All: Whether the dot character should match all characters including newlines (default: yes)

### Outputs
- Positive: Data that is output when a pattern match is found
- Negative: Data that is output when no pattern match is found

### Possible use case
Filtering customer feedback messages to route complaints (containing words like "unhappy" or "disappointed") to the customer service team, and positive feedback to the marketing team.

## Extract Text Information

### What it is
A tool that pulls out specific pieces of information from text based on patterns.

### What it does
Searches through text to find and extract specific portions that match a given pattern.

### How it works
The block searches for text matching a specified pattern and extracts either the entire match or a specific part of it. It can find either the first match or all matches, depending on the settings.

### Inputs
- Text: The text to extract information from
- Pattern: The pattern (in regular expression format) to use for extraction
- Group: Which portion of the match to extract (default: 0 for entire match)
- Case Sensitive: Whether to consider upper/lowercase differences (default: yes)
- Dot All: Whether the dot character should match all characters including newlines (default: yes)
- Find All: Whether to find all matches or just the first one (default: no)

### Outputs
- Positive: The extracted text when matches are found
- Negative: The original text when no matches are found

### Possible use case
Extracting email addresses or phone numbers from a document or extracting product codes from customer messages.

## Fill Text Template

### What it is
A tool that creates customized text by filling in a template with specific values.

### What it does
Takes a template and a set of values, then creates a final text by replacing placeholders in the template with the actual values.

### How it works
The block uses a template with special markers ({{placeholder}}) and replaces these markers with corresponding values from a provided dictionary.

### Inputs
- Values: A dictionary of names and values to be used in the template
- Format: The template text with placeholders for the values

### Outputs
- Output: The final text with all placeholders replaced with their corresponding values

### Possible use case
Creating personalized email messages by filling in customer names, order numbers, and other specific details in a standard template.

## Combine Texts

### What it is
A tool that joins multiple pieces of text into a single text.

### What it does
Takes a list of separate text pieces and combines them into one unified text, optionally placing a delimiter between each piece.

### How it works
The block takes multiple text inputs and joins them together, with an optional separator between each piece of text.

### Inputs
- Input: A list of text pieces to be combined
- Delimiter: The text to insert between each piece (default: nothing)

### Outputs
- Output: The final combined text

### Possible use case
Combining separate paragraphs into a single document, or joining pieces of an address with appropriate spacing or punctuation.

