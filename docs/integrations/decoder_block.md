# Text Decoder

## What it is
A tool that converts text with special characters into regular, readable text.

## What it does
It takes a string of text that contains escaped characters (like '\n' for new lines or '\"' for quotation marks) and converts them into their actual representations in the text.

## How it works
The Text Decoder looks at the input text and identifies special character sequences. It then replaces these sequences with their actual characters, making the text more readable and removing any escape characters.

## Inputs
| Input | Description |
|-------|-------------|
| Text | The text you want to decode, which may contain escaped characters like '\n' for new lines or '\"' for quotation marks |

## Outputs
| Output | Description |
|--------|-------------|
| Decoded Text | The text after processing, with all escape sequences converted to their actual characters |
| Error | If there's a problem during the decoding process, an error message will be provided instead |

## Possible use case
Imagine you receive a text message that looks like this: "Hello\nWorld!\nThis is a \"quoted\" string." The Text Decoder can convert it into a more readable format:

```
Hello
World!
This is a "quoted" string.
```

This could be useful when working with data from various sources where text might be encoded to preserve special characters, such as when importing data from a file or receiving it from an API.

# Text Encoder

## What it is
A tool that converts regular text into text with escape sequences for special characters.

## What it does
It takes a string of text and escapes special characters (like newlines `\n`, tabs `\t`, or quotes `\"`) so that it can be safely stored or transmitted. This is the reverse operation of the Text Decoder.

## Inputs
| Input | Description |
|-------|-------------|
| Text | The text you want to encode, which may contain newlines or special characters |

## Outputs
| Output | Description |
|--------|-------------|
| Encoded Text | The text after processing, with special characters replaced by escape sequences |
| Error | If there's a problem during the encoding process, an error message will be provided instead |

## Possible use case
If you need to store multiline text in a single line format (like a CSV or JSON string), you can use the Text Encoder.

Input:
```
Hello
World!
"Quoted"
```

Output:
`Hello\nWorld!\n\"Quoted\"`