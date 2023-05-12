# Read the docs

An AI agent needs to be capable of retrieving information and reading the manual or documentation about a topic before executing actions that could potentially waste time and incur costs. By accessing and understanding relevant information, the agent can make informed decisions and avoid unnecessary or inefficient actions. In the case of an LLM like GPT, this capability becomes particularly important as it helps prevent the misuse or unnecessary consumption of valuable resources, such as API tokens that incur costs. By retrieving information and reading the necessary documentation, the agent can gain insights, clarify requirements, understand limitations, and ensure optimal utilization of resources, ultimately enhancing its efficiency, cost-effectiveness, and overall performance.

## Description

In this challenge, participants will develop an AI agent capable of executing CLI tools based on their help verbose output. The agent must analyze the help output, learn the required parameters and their usage, and accurately execute the tool by providing the necessary parameters. Success in the challenge will be determined by the agent's ability to accurately interpret the help verbose output, identify the required parameters, and execute the CLI tool accordingly. The challenge aims to assess the agent's capability to leverage textual information to execute CLI tools effectively, showcasing its ability to adapt and utilize tool documentation for successful execution.


## Idea
A single pytest-based unit test can be used to support different help verbose strings by parameterizing the test. This approach allows multiple sets of help verbose strings to be provided as input to the test function, enabling the agent's behavior to be evaluated across various scenarios. Each set of help verbose strings represents a different CLI tool or version, containing unique parameter information.

By parameterizing the test, the agent can be evaluated for its capability to adapt its behavior based on the specific help verbose string it receives. The test function can compare the agent's output or behavior against expected outcomes for each set of help verbose strings. This evaluation helps determine the agent's ability to interpret and utilize the provided help verbose strings accurately, extract the necessary parameter information, and adjust its behavior accordingly.

The parameterized unit test not only supports different help verbose strings but also serves as a mechanism to evaluate the agent's adaptability. By assessing the agent's behavior across various scenarios, it becomes possible to gauge its flexibility, agility, and capability to handle different inputs effectively. This approach allows for a comprehensive evaluation of the agent's ability to adapt to different help verbose strings and adjust its behavior accordingly.


## Scope

The challenge focuses on designing an AI agent capable of executing CLI tools based on varying help verbose strings. Participants will create a pytest-based unit test that supports different sets of help verbose strings, parameterizing the test to evaluate the agent's adaptability. The agent's behavior will be assessed by comparing its output or behavior against expected outcomes for each set of help verbose strings. The challenge aims to evaluate the agent's capability to interpret and utilize different help verbose strings accurately, extract necessary parameter information, and adapt its behavior accordingly. It focuses on assessing the agent's adaptability and flexibility in handling diverse CLI tool inputs, showcasing its ability to adjust its behavior based on the specific requirements of each tool.

## Success Evaluation
The success of the challenge hinges on the agent's ability to effectively utilize the help verbose output to guide its use of the CLI tool. Evaluation will focus on the agent's accurate identification and usage of parameters, effective error handling, and adaptability to different help verbose strings. The challenge aims to assess the agent's proficiency in leveraging the provided information to execute CLI tools accurately and efficiently.

