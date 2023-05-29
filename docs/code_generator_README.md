# 😎 CodeGenerator

The `CodeGenerator` class is your ultimate code generation companion! 🚀 It harnesses the power of pre-trained language models to generate code snippets with ease. Whether you need a function signature, a comment, a docstring, or even a fill-in-the-middle prompt, the `CodeGenerator` has got you covered! 💡💻

This project is inspired by the awesome work done in [the-crypt-keeper/tiny_starcoder](https://github.com/the-crypt-keeper/tiny_starcoder). Credit goes to them for the original idea!

## Installation

To embark on this exciting code generation journey, make sure you have the following dependencies installed:

- Python 3.6 or above 🐍
- Hugging Face Transformers library 🤗
- AutoGPT library 🤖
- Dotenv library 🌌

You can install these dependencies by running the following command:

```shell
pip install transformers autogpt python-dotenv
```

Now, let's dive into some cool use cases where the `CodeGenerator` can save the day! 🦸‍♂️🦸‍♀️

## Use Cases

**Function Signature**

```python
generated_code = code_generator._generate_function_signature(signature)
```

**Comment**

```python
generated_code = code_generator._generate_comment(comment)
```

**Docstring**

```python
generated_code = code_generator._generate_docstring(docstring)
```

**Fill in the Middle**

```python
generated_code = code_generator._generate_fill_in_the_middle(input_text)
```

## Prompt Styles

**Prompt Style 1: Function Signature**

This is not a multi-billion parameter chat model, it's a tiny specialized code generation model and has to be prompted correctly.

The simplest possible prompt is a function signature:

```python
def print_hello_world():
```

This returns:

```python
def print_hello_world():
    """Prints hello world"""

    print("Hello World!")


if __name__ == "__main__":
    main()
```

Notes:

- `<fim_prefix>` is a special character for this model
- It frequently puts in nonsense main or test code after what has been requested

**Prompt Style 2: A Comment**

Another possible prompt style is a comment that describes what this function does:

```python
# a python function that says hello
```

This returns:

```python
def say_hello():
    print("Hello World!")

if __name__ == "__main__":
    say_hello()
```

Note:

- This style can be combined with Style 1, you can give both a comment and a function signature.

**Prompt Style 3: A Docstring**

This is Python after all, so this model understands docstrings:

```python
""" a python function that says hello """
```

This returns:

```python
""" a python function that says hello """
def say_hello():
    print("Hello World!")


Note:

- This prompt style did not elicit the post-generation garbage; this is worth exploring further.

**Prompt Style 4: Fill in the Middle**

```python
<fim_prefix>def print_one_two_three():
    print('one')
    <fim_suffix>
    print('three')<fim_middle>
```

Returns:

```python
<fim_prefix>def print_one_two_three():
    print('one')
    <fim_suffix>
    print('three')<fim_middle>print('two')
```

The model filled in the 'two' in the middle.
```

## Error Handling

If an error occurs during code generation, an exception will be raised with a descriptive error message. The traceback will also be printed for debugging purposes.

## Available Commands

The `CodeGenerator` class provides the following command functions that can be used with the AutoGPT library:

- `generate_code_signature(input_text: str, config: 'Config') -> str`: Generate code using a function signature as the input text.
- `generate_code_comment(input_text: str, config: 'Config') -> str`: Generate code using a comment as the input text.
- `generate_code_docstring(input_text: str, config: 'Config') -> str`: Generate code using a docstring as the input text.
- `generate_code_fill_in(input_text: str, config: 'Config') -> str`: Generate code using a "fill in the middle" input text.

These commands can be invoked using the AutoGPT library's command-line interface.

## Notes

- The `CodeGenerator` class assumes that you have set the necessary environment variables using a `.env` file or other means supported by the `dotenv` library.

- The code generation process may take some time depending on the complexity of the generated code and the computational resources available.

Now, go ahead and unleash the power of code generation with the `CodeGenerator`! 🚀🔥
```
```
