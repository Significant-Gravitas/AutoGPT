from autogpt.llm_utils import create_chat_completion


def summarize_text(driver, text, question):
    if not text:
        return "Error: No text to summarize"

    text_length = len(text)
    print(f"Text length: {text_length} characters")

    summaries = []
    chunks = list(split_text(text))

    scroll_ratio = 1 / len(chunks)
    for i, chunk in enumerate(chunks):
        scroll_to_percentage(driver, scroll_ratio * i)
        print(f"Summarizing chunk {i + 1} / {len(chunks)}")
        messages = [create_message(chunk, question)]

        summary = create_chat_completion(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,
        )
        summaries.append(summary)

    print(f"Summarized {len(chunks)} chunks.")

    combined_summary = "\n".join(summaries)
    messages = [create_message(combined_summary, question)]

    return create_chat_completion(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
    )


def split_text(text, max_length=8192):
    paragraphs = text.split("\n")
    current_length = 0
    current_chunk = []

    for paragraph in paragraphs:
        if current_length + len(paragraph) + 1 <= max_length:
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 1
        else:
            yield "\n".join(current_chunk)
            current_chunk = [paragraph]
            current_length = len(paragraph) + 1

    if current_chunk:
        yield "\n".join(current_chunk)


def create_message(chunk, question):
    return {
        "role": "user",
        "content": f'"""{chunk}""" Using the above text, please answer the following'
        f' question: "{question}" -- if the question cannot be answered using the text,'
        " please summarize the text.",
    }


def scroll_to_percentage(driver, ratio):
    if ratio < 0 or ratio > 1:
        raise ValueError("Percentage should be between 0 and 1")
    driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight * {ratio});")
