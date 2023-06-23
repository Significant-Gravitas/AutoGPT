# Challenges Data Schema of Benchmark

## General challenges

Input:

- **category** (str): information-retrieval
- **difficulty**(str): the difficulty of this query. choices from

## Information-retrieval challenges

Input:

- **category** (str): information-retrieval
- **task** (str): the question the agent needs to be solve.
- **ground** (dict): The ground truth.
  - **answer** (str): The raw text of ground truth answer
  - **should_contain** (list): the exact strings that is required in the final answer
  - **should_not_contain** (list): the exact strings that should not be in the final answer
  - **files**: files that the are used for retrieval. Can specify file here or an extension **TODO:** like .txt
- **difficulty**(str): the difficulty of this query. choices from
- **mock_func**: function to mock the agent's response. This is used for testing purposes

Example:

```python
{
  "category": "retrieval",
  "task": "What is the capital of America?",
  "ground": {
    "answer": "Washington",
    "should_contain": ["Washington"],
    "should_not_contain": ["New York", "Los Angeles", "San Francisco"],
    "files": ["file_to_check.txt"]
  },
  "difficulty": "easy"
}

```

Output:

- **score** (float): scores range from [0, 1]
