## `split_text`

```python
split_text(text: str, max_length: int = CFG.browse_chunk_max_length, model: str = CFG.fast_llm_model, question: str = "") -> Generator[str, None, None]
```

Split text into chunks of a maximum length.

**Args**:

- `text`: The text to split.
- `max_length` (optional, default=8192): The maximum length of each chunk.
- `model` (optional, default=CFG.fast_llm_model): The language model to use.
- `question` (optional, default=""): The question to ask the model.

**Returns**:

A generator that returns the next chunk of text.

**Raises**:

- `ValueError`: If the text is longer than the maximum length.

---

## `token_usage_of_chunk`

```python
token_usage_of_chunk(messages, model)
```

Counts the number of tokens used in a chunk.

**Args**:

- `messages`: The messages to count the tokens of.
- `model`: The language model to use.

**Returns**:

The number of tokens used in the messages.

---

## `summarize_text`

```python
summarize_text(url: str, text: str, question: str, driver: Optional[WebDriver] = None) -> str
```

Summarize text using the OpenAI API.

**Args**:

- `url`: The url of the text.
- `text`: The text to summarize.
- `question`: The question to ask the model.
- `driver` (optional): The webdriver to use to scroll the page.

**Returns**:

The summary of the text.

---

## `scroll_to_percentage`

```python
scroll_to_percentage(driver: WebDriver, ratio: float) -> None
```

Scroll to a percentage of the page.

**Args**:

- `driver`: The webdriver to use.
- `ratio`: The percentage to scroll to.

**Raises**:

- `ValueError`: If the ratio is not between 0 and 1.

---

## `create_message`

```python
create_message(chunk: str, question: str) -> Dict[str, str]
```

Create a message for the chat completion.

**Args**:

- `chunk`: The chunk of text to summarize.
- `question`: The question to answer.

**Returns**:

The message to send to the chat completion.