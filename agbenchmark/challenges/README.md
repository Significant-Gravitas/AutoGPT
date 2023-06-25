# Challenges Data Schema of Benchmark

## General challenges

Input:

- **category** (str[]): Category of the challenge such as 'retrieval', 'comprehension', etc. _this is not currently used. for the future it may be needed_
- **task** (str): The task that the agent needs to solve.
- **dependencies** (str[]): The dependencies that the challenge needs to run. Needs to be the full node to the test function.
- **ground** (dict): The ground truth.
  - **answer** (str): The raw text of the ground truth answer.
  - **should_contain** (list): The exact strings that are required in the final answer.
  - **should_not_contain** (list): The exact strings that should not be in the final answer.
  - **files** (list): Files that are used for retrieval. Can specify file here or an extension.
- **mock_func** (str): Function to mock the agent's response. This is used for testing purposes.
- **info** (dict): Additional info about the challenge.
  - **difficulty** (str): The difficulty of this query.
  - **description** (str): Description of the challenge.
  - **side_effects** (str[]): Describes the effects of the challenge.

Example:

```python
{
  "category": ["basic"],
  "task": "Write the string 'random string' before any existing text to the file called file_to_check.txt",
  "dependencies": [
    "agbenchmark/tests/basic_abilities/write_file/write_file_test.py::TestWriteFile::test_write_file"
  ],
  "ground": {
    "answer": "random string: this is how we're doing",
    "should_contain": ["random string: this is how we're doing"],
    "files": ["file_to_check.txt"]
  },
  "mock_func": "basic_read_file_mock",
  "info": {
    "description": "This reads the file quickly",
    "difficulty": "basic",
    "side_effects": [""]
  }
}

```

Current Output:

- **score** (float): scores range from [0, 1]
