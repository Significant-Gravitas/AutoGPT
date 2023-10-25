# Steps Module
The steps module defines a series of steps that the AI can perform to generate code, clarify instructions, generate specifications, and more. Each step is a function that takes an AI and a set of databases as arguments and returns a list of messages. The steps are defined in the `gpt_engineer/steps.py` file.

<br>

## Steps
Here are the steps defined in the steps module:

`setup_sys_prompt(dbs)`: This function sets up the system prompt by combining the generate preprompt and the philosophy preprompt.

`simple_gen(ai: AI, dbs: DBs)`: This function runs the AI on the main prompt and saves the results.

`clarify(ai: AI, dbs: DBs)`: This function asks the user if they want to clarify anything and saves the results to the workspace.

`gen_spec(ai: AI, dbs: DBs)`: This function generates a spec from the main prompt + clarifications and saves the results to the workspace.

`respec(ai: AI, dbs: DBs)`: This function asks the AI to review a specification for a new feature and give feedback on it.

`gen_unit_tests(ai: AI, dbs: DBs)`: This function generates unit tests based on the specification.

`gen_clarified_code(ai: AI, dbs: DBs)`: This function generates code based on the main prompt and clarifications.

`gen_code(ai: AI, dbs: DBs)`: This function generates code based on the specification and unit tests.

`execute_entrypoint(ai, dbs)`: This function executes the entrypoint of the generated code.

`gen_entrypoint(ai, dbs)`: This function generates the entrypoint for the generated code.

`use_feedback(ai: AI, dbs: DBs)`: This function uses feedback from the user to improve the generated code.

`fix_code(ai: AI, dbs: DBs)`: This function fixes any errors in the generated code.

<br>

## Configurations
Different configurations of steps are defined in the STEPS dictionary. Each configuration is a list of steps that are run in order.

The available configurations are:
```python
DEFAULT: clarify, gen_clarified_code, gen_entrypoint, execute_entrypoint
BENCHMARK: simple_gen, gen_entrypoint
SIMPLE: simple_gen, gen_entrypoint, execute_entrypoint
TDD: gen_spec, gen_unit_tests, gen_code, gen_entrypoint, execute_entrypoint
TDD_PLUS: gen_spec, gen_unit_tests, gen_code, fix_code, gen_entrypoint, execute_entrypoint
CLARIFY: clarify, gen_clarified_code, gen_entrypoint, execute_entrypoint
RESPEC: gen_spec, respec, gen_unit_tests, gen_code, fix_code, gen_entrypoint, execute_entrypoint
USE_FEEDBACK: use_feedback, gen_entrypoint, execute_entrypoint
EXECUTE_ONLY: execute_entrypoint
```

<br>

## Conclusion
The steps module provides a flexible framework for running different sequences of steps in the GPT-Engineer system. Each step is a function that performs a specific task, such as generating code, clarifying instructions, or executing the generated code. The steps can be combined in different configurations to achieve different outcomes.
