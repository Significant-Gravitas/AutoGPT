# Text Encoder

## What it is
A tool that converts text containing special characters into escaped text sequences.

## What it does
It takes a string of text that contains special characters (like new lines or quotation marks) and converts them into their escape sequence representations (like '\n' for new lines).

## How it works
The Text Encoder takes the input string and applies Python's `unicode_escape` encoding (equivalent to `codecs.encode(text, "unicode_escape").decode("utf-8")`) to transform special characters like newlines, tabs, and backslashes into their escaped forms.

The block relies on the input schema to ensure the value is a string; non-string inputs are rejected by validation, and any encoding failures surface as block errors. Non-ASCII characters are emitted as `\uXXXX` sequences, which is useful for ASCII-only payloads.

## Inputs

| Input | Description |
|-------|-------------|
| Text | The text you want to encode, which may contain special characters like new lines or quotation marks |

## Outputs

| Output | Description |
|--------|-------------|
| Encoded Text | The text after processing, with all special characters converted to their escape sequences |

## Use Case
- **JSON payload preparation:** Encode multiline or quoted text before embedding it in JSON string fields.
- **Config/ENV generation:** Convert template text into escaped strings for `.env` or YAML values.
- **Snapshot fixtures:** Produce stable escaped strings for golden files or API tests.

## Example
Imagine you have a piece of text with line breaks that you need to store in a JSON file or send through an API:

```text
Hello
World!
This is a "quoted" string.
```

The Text Encoder can convert it into:

```text
Hello\nWorld!\nThis is a "quoted" string.
```

This is useful when you need to prepare text for storage in formats that require escape sequences, or when sending data to systems that expect encoded text. It's the inverse operation of the Text Decoder block.
