# Memory Challenge B

**Status**: Current level to beat: level 3

**Command to try**: 

```shell
pytest -s tests/challenges/memory/test_memory_challenge_b.py --level=3
```

## Description

The agent, Follow-Instructions-GPT, has the following goals:

- Use the command `read_file` to read the `instructions_1.txt` file.
- Follow the instructions in the `instructions_1.txt` file.

The challenge involves a series of text files containing instructions and task IDs. The agent must follow the instructions and keep track of the task IDs as it progresses through the files.

## Files

- `instructions_1.txt`

The current task_id is 4563.\n[NOISE intended to confuse the agent]
Read the file instructions_2.txt using the read_file command.


- `instructions_2.txt`

The current task_id is 6182.\n[NOISE intended to confuse the agent]
Read the file instructions_3.txt using the read_file command.


...and so on.

- `instructions_n.txt`

The current task_id is 8912.
Write all the task_ids into the file output.txt. The file has not been created yet. After that, use the task_complete command.


## Objective

The objective of this challenge is to test the agent's ability to follow instructions and maintain memory of the task IDs throughout the process. The agent successfully completed this challenge if it wrote the task ids in a file.
