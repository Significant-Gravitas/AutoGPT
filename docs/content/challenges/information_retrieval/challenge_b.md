# Information Retrieval Challenge B

**Status**: Beaten

**Command to try**:

```
pytest -s tests/challenges/information_retrieval/test_information_retrieval_challenge_b.py
```

## Description

The agent's goal is to find the names, affiliated university, and discovery of the individuals who won the nobel prize for physics in 2010.

It should write the result in a file called 2010_nobel_prize_winners.txt.

The agent should be able to beat this test consistently (this is the hardest part).

## Objective

The objective of this challenge is to test the agent's ability to retrieve multiple pieces of related information in a consistent way.
The agent should not use google to perform the task, because it should already know the answer. This why the task fails after 2 cycles (1 cycle to retrieve information, 1 cycle to write the file)
