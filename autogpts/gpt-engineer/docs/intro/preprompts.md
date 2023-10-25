# Preprompts

The preprompts are a set of predefined prompts that guide the AI in performing different tasks. They are stored as text files in the `gpt_engineer/preprompts` directory.

<br>

### 1. Fix Code (`gpt_engineer/preprompts/fix_code`)

This prompt instructs the AI to fix a program and make it work according to the best of its knowledge. The AI is expected to provide fully functioning, well-formatted code with few comments, that works and has no bugs.

<br>

### 2. Generate (`gpt_engineer/preprompts/generate`)

This prompt instructs the AI to generate code based on a set of instructions. The AI is expected to think step by step and reason itself to the right decisions to make sure it gets it right. It should first lay out the names of the core classes, functions, methods that will be necessary, as well as a quick comment on their purpose. Then it should output the content of each file including ALL code.

<br>

### 3. Philosophy (`gpt_engineer/preprompts/philosophy`)

This prompt provides the AI with a set of best practices to follow when writing code. For example, it instructs the AI to always put different classes in different files, to always create an appropriate requirements.txt file for Python, and to always follow the best practices for the requested languages in terms of describing the code written as a defined package/project.

<br>

### 4. QA (`gpt_engineer/preprompts/qa`)

This prompt instructs the AI to read instructions and seek to clarify them. The AI is expected to first summarise a list of super short bullets of areas that need clarification. Then it should pick one clarifying question, and wait for an answer from the user.

<br>

### 5. Respec (`gpt_engineer/preprompts/respec`)

This prompt instructs the AI to review a specification for a new feature and give feedback on it. The AI is expected to identify anything that might not work the way intended by the instructions, anything missing for the program to work as expected, and anything that can be simplified without significant drawback.

<br>

### 6. Spec (`gpt_engineer/preprompts/spec`)

This prompt instructs the AI to make a specification for a program. The AI is expected to be super explicit about what the program should do, which features it should have, and give details about anything that might be unclear.

<br>

### 7. Unit Tests (`gpt_engineer/preprompts/unit_tests`)

This prompt instructs the AI to write tests according to a specification using Test Driven Development. The tests should be as simple as possible, but still cover all the functionality.

<br>

### 8. Use Feedback (`gpt_engineer/preprompts/use_feedback`)
This prompt instructs the AI to generate code based on a set of instructions and feedback defined by the user.

<br>

## Conclusion

GPT-Engineer provides a powerful tool for automating software engineering tasks using GPT-4. It includes a flexible framework for running different sequences of steps, each guided by a set of predefined prompts. The AI is expected to follow best practices and reason itself to the right decisions to ensure high-quality code generation.
