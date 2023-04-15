[View code on GitHub](https://github.com/Significant-Gravitas/Auto-GPT/.autodoc/docs/json/autogpt/processing)

The `.autodoc/docs/json/autogpt/processing` folder contains essential text processing functions for the Auto-GPT project. These functions handle text data, summarize content, interact with web pages, and create chat completion messages.

The `text.cpython-39.pyc` file contains four main functions:

1. `split_text(text: str, max_length: int = 8192) -> Generator[str, None, None]`: This function takes a string `text` and an optional `max_length` (default 8192) as input. It splits the text into chunks of maximum length and yields each chunk. If the text is longer than the maximum length, a `ValueError` is raised. This function can be used to process large text inputs and ensure they fit within the constraints of the OpenAI API.

Example usage:

```python
for chunk in split_text(long_text, max_length=4096):
    process_chunk(chunk)
```

2. `summarize_text(url: str, text: str, question: str, driver: WebDriver) -> str`: This function takes a `url`, `text`, `question`, and a `WebDriver` instance as input. It summarizes the text using the OpenAI API and returns the summary as a string. This function can be used to generate summaries of web page content or other text data.

Example usage:

```python
summary = summarize_text("https://example.com/article", article_text, "What is the main point of the article?", driver)
```

3. `scroll_to_percentage(driver: WebDriver, ratio: float) -> None`: This function takes a `WebDriver` instance and a `ratio` (float between 0 and 1) as input. It scrolls the webpage to the specified percentage. A `ValueError` is raised if the ratio is not between 0 and 1. This function can be used to interact with web pages and ensure that relevant content is visible on the screen.

Example usage:

```python
scroll_to_percentage(driver, 0.5)  # Scroll to 50% of the webpage
```

4. `create_message(chunk: str, question: str) -> Dict[str, str]`: This function takes a `chunk` of text and a `question` as input. It returns a dictionary with the message to send to the chat completion. This function can be used to format text data and questions for use with the OpenAI API.

Example usage:

```python
message = create_message(text_chunk, "What is the main idea?")
```

These functions work together to process and summarize text, interact with the OpenAI API, and handle web page scrolling in the larger Auto-GPT project.
