from autogpt.llm.chat import create_chat_message


def test_create_chat_message():
    """Test that the function returns a dictionary with the correct keys and values when valid strings are provided for role and content."""
    result = create_chat_message("system", "Hello, world!")
    assert result == {"role": "system", "content": "Hello, world!"}
