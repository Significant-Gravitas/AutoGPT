
## Text Decoder

### What it is
A utility block that converts text containing special escape sequences into their proper readable format.

### What it does
This block takes text that contains special characters written in their "escaped" form (like \n for new lines or \" for quotation marks) and converts them into their actual, readable representations. It's like translating computer-friendly text into human-friendly text.

### How it works
The block processes the input text by identifying special character sequences (escape sequences) and replacing them with their actual characters. For example, when it sees '\n', it creates an actual line break in the text, and when it sees '\"', it creates an actual quotation mark.

### Inputs
- Text: The input text that contains escaped characters. This could be any text string that has special characters written in their escaped form (like \n for new lines or \" for quote marks).

### Outputs
- Decoded Text: The processed text where all escape sequences have been converted to their actual characters. The output is more readable and properly formatted text with actual line breaks, quotation marks, and other special characters.

### Possible use case
A content editor receives a text file from a technical system that contains escaped characters. Instead of manually replacing all these special characters, they can use this Text Decoder block to automatically convert the text into a properly formatted version. For example:

Input: "Hello\nWorld!\nThis is a \"quoted\" string."
Would become:
Hello
World!
This is a "quoted" string.

This is particularly useful when working with text exported from programming environments or when processing system-generated text that needs to be made human-readable.
