# OpenAI Type Helpers
This module contains a `TypedDict` subclass for the `OpenAI` library's `Message` object.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)

## Installation
This module requires no additional installation beyond having the `typing` module installed, which should come pre-installed with Python 3.5+.

## Usage
Import the `TypedDict` module from the `typing` module and the `TypedDict` subclass `Message` from this module.

```python
from typing import TypedDict
from openai_helpers import Message
```

## Examples
Create an instance of `Message` using the `role` and `content` attributes.

```python
message = Message(role="user", content="Hello, world!")
```

The `message` variable would now be of type `Message` and have attributes `role` equal to `"user"` and `content` equal to `"Hello, world!"`.