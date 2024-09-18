# Memory Challenge D

**Status**: Current level to beat: level 1

**Command to try**: 

```shell
pytest -s tests/challenges/memory/test_memory_challenge_d.py --level=1
```

## Description

The provided code is a unit test designed to validate an AI's ability to track events and beliefs of characters in a story involving moving objects, specifically marbles. This scenario is an advanced form of the classic "Sally-Anne test", a psychological test used to measure a child's social cognitive ability to understand that others' perspectives and beliefs may differ from their own.

Here is an explanation of the challenge:

The AI is given a series of events involving characters Sally, Anne, Bob, and Charlie, and the movements of different marbles. These events are designed as tests at increasing levels of complexity.

For each level, the AI is expected to keep track of the events and the resulting beliefs of each character about the locations of each marble. These beliefs are affected by whether the character was inside or outside the room when events occurred, as characters inside the room are aware of the actions, while characters outside the room aren't.

After the AI processes the events and generates the beliefs of each character, it writes these beliefs to an output file in JSON format.

The check_beliefs function then checks the AI's beliefs against the expected beliefs for that level. The expected beliefs are predefined and represent the correct interpretation of the events for each level.

If the AI's beliefs match the expected beliefs, it means the AI has correctly interpreted the events and the perspectives of each character. This would indicate that the AI has passed the test for that level.

The test runs for levels up to the maximum level that the AI has successfully beaten, or up to a user-selected level.


## Files

- `instructions_1.txt`

```
Sally has a marble (marble A) and she puts it in her basket (basket S), then leaves the room. Anne moves marble A from Sally's basket (basket S) to her own basket (basket A).
```


- `instructions_2.txt`

```
Sally gives a new marble (marble B) to Bob who is outside with her. Bob goes into the room and places marble B into Anne's basket (basket A). Anne tells Bob to tell Sally that he lost the marble b. Bob leaves the room and speaks to Sally about the marble B. Meanwhile, after Bob left the room, Anne moves marble A into the green box, but tells Charlie to tell Sally that marble A is under the sofa. Charlie leaves the room and speak to Sally about the marble A as instructed by Anne.
```

...and so on.

- `instructions_n.txt`

The expected believes of every characters are given in a list:

```json
expected_beliefs = {
    1: {
        'Sally': {
            'marble A': 'basket S',
        },
        'Anne': {
            'marble A': 'basket A',
        }
    },
    2: {
        'Sally': {
            'marble A': 'sofa',  # Because Charlie told her
        },
        'Anne': {
            'marble A': 'green box',  # Because she moved it there
            'marble B': 'basket A',  # Because Bob put it there and she was in the room
        },
        'Bob': {
            'B': 'basket A',  # Last place he put it
        },
        'Charlie': {
            'A': 'sofa',  # Because Anne told him to tell Sally so
        }
    },...
```

## Objective

This test essentially checks if an AI can accurately model and track the beliefs of different characters based on their knowledge of events, which is a critical aspect of understanding and generating human-like narratives. This ability would be beneficial for tasks such as writing stories, dialogue systems, and more.
