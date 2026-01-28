# Text Encoder

## What it is
A tool that converts text containing special characters into escaped text sequences.

## What it does
It takes a string of text that contains special characters (like new lines or quotation marks) and converts them into their escape sequence representations (like '\n' for new lines).

## How it works
The Text Encoder examines the input text and identifies special characters. It then replaces these characters with their escape sequence equivalents, making the text safe for storage or transmission in formats that don't support raw special characters.

## Inputs
| Input | Description |
|-------|-------------|
| Text | The text you want to encode, which may contain special characters like new lines or quotation marks |

## Outputs
| Output | Description |
|--------|-------------|
| Encoded Text | The text after processing, with all special characters converted to their escape sequences |

## Possible use case
Imagine you have a piece of text with line breaks that you need to store in a JSON file or send through an API:

```
Hello
World!
This is a "quoted" string.
```

The Text Encoder can convert it into:

```
Hello\nWorld!\nThis is a "quoted" string.
```

This is useful when you need to prepare text for storage in formats that require escape sequences, or when sending data to systems that expect encoded text. It's the inverse operation of the Text Decoder block.
