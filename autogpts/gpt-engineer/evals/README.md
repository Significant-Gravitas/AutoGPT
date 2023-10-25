

# Evals

Evals are a set of tests that allow us to measure the performance of the gpt-engineer whole system.  This includes the gpt-enginer code, options and the chosen LLM.

### Running Evals

To run the existing code evals make sure you are in the gpt-engineer top level directory (you should see a directory called `evals`) type:

`python evals/evals_existing_code.py`  This will run the default test file: `evals/existing_code_eval.yaml`, or you can run any YAML file of tests you wish with the command: `python evals/evals_existing_code.py your_test_file.yaml`

Similarly to run the new code evals type:

`python evals/evals_new_code.py`
