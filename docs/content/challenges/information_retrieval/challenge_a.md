# Information Retrieval Challenge A

**Status**: Current level to beat: level 2

**Command to try**:

```
pytest -s tests/challenges/information_retrieval/test_information_retrieval_challenge_a.py --level=2
```

## Description

The agent's goal is to find the revenue of Tesla:
- level 1 asks the revenue of Tesla in 2022 and explicitly asks to search for 'tesla revenue 2022'
- level 2 is identical but doesn't ask to search for 'tesla revenue 2022'
- level 3 asks for tesla's revenue by year since its creation.

It should write the result in a file called output.txt.

The agent should be able to beat this test consistently (this is the hardest part).
## Objective

The objective of this challenge is to test the agent's ability to retrieve information in a consistent way.
