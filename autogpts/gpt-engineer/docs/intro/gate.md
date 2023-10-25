# GPT-Engineer

GPT-Engineer is an open-source project that leverages the capabilities of OpenAI's GPT-4 model to automate various software engineering tasks. It is designed to interact with the GPT-4 model in a conversational manner, guiding the model to generate code, clarify instructions, generate specifications, and more. The project is built with a modular architecture, making it easy to extend and customize for various use cases.

## Core Components

GPT-Engineer is composed of several core components that work together to provide its functionality:

- **AI Class**: The AI class serves as the main interface to the GPT-4 model. It provides methods to start a conversation with the model, continue an existing conversation, and format system and user messages.

- **DB Class**: The DB class represents a simple database that stores its data as files in a directory. It is a key-value store, where keys are filenames and values are file contents.

- **Steps Module**: The steps module defines a series of steps that the AI can perform to generate code, clarify instructions, generate specifications, and more. Each step is a function that takes an AI and a set of databases as arguments and returns a list of messages.

## Usage

GPT-Engineer is designed to be easy to use, even for users without a background in coding. Users can write prompts in plain English, and GPT-Engineer will guide the GPT-4 model to generate the desired output. The generated code is saved as files in a workspace, and can be executed independently of the GPT-Engineer system.

## Development and Community

GPT-Engineer is an open-source project, and contributions from the community are welcomed and encouraged. The project is hosted on GitHub, where users can report issues, suggest enhancements, and contribute code.

## See Also

- [User Guide](#user-guide)
- [AI Class Documentation](#ai-class)
- [DB Class Documentation](#db-class)
- [Steps Module Documentation](#steps-module)
- [Harmony of AI, DB, and Steps](#harmony-of-ai-db-and-steps)
- [Chat Parsing & Self Code Execution](#chat_to_files.py)

## References

- [GPT-Engineer GitHub Repository](https://github.com/AntonOsika/gpt-engineer)
