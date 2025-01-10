
## Text Decoder

### What it is
A utility block that converts text containing escaped characters into readable, properly formatted text.

### What it does
This block takes text that contains special escape sequences (like \n for new lines or \" for quotation marks) and converts them into their actual intended characters. It makes text that might look messy or coded into clean, properly formatted text with actual line breaks and proper punctuation.

### How it works
The block processes the input text and looks for special character combinations (escape sequences) that represent formatting or special characters. It then replaces these sequences with their actual intended characters. For example, it turns "\n" into actual line breaks and '\"' into regular quotation marks.

### Inputs
- Text: The text you want to decode, which may contain escape sequences like \n for new lines or \" for quotation marks. This could be text that was copied from a programming environment or exported from another system.

### Outputs
- Decoded Text: The cleaned-up version of your text, with all escape sequences properly converted into their actual characters. The text will appear properly formatted with actual line breaks and proper punctuation.

### Possible use case
Imagine you're working with text exported from a programming environment or database, and it looks like this: "Hello\nWorld!\nThis is a \"quoted\" string." The Text Decoder would transform this into properly formatted text that looks like this:
```
Hello
World!
This is a "quoted" string.
```
This would be particularly useful when processing exported data, working with system outputs, or cleaning up text that contains escape sequences for display to end users.

