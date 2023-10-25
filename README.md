# Ghostcoder - AutoGPT style
This is a submission to the AutoGPT Arena Hacks hackathon. The Agent code is in the directory `autogpts/ghostcoder`.

Ghostcoder is an agent specialized in writing, verifying, and editing code. To accomplish this functionality
from the experimental project [Ghostcoder](https://github.com/aorwall/ghostcoder) is used. 

## Concepts

### Write code
When a task comes in related to writing code, the "write code" ability will be executed.

This is managed with a prompt instructing GPT to demonstrate its understanding of the task by listing the requirements.

Any other files in the workspace will be included in the prompt unless they exceed the maximum token count.

The code is then written in the following format:

/file.py
```python
# ... code  
```

The file is then saved in the agent's workspace.

This simple approach works quite robustly on all benchmarks except `Battleship` (and occasionally 
`FileOrganizer` due to the file types GPT interprets as a "document").

### Verify code
If there are tests linked to a task, they will be run after the code has been written.
The results of these are parsed into a more concise format to reduce context length.

#### Exempel
The following output is from `pytest`:

```
__________________________________ test_cant_place_ship_after_all_ships_placed ___________________________________

battleship_game = <battleship.Battleship object at 0x7f8bbcd1b8d0>
initialized_game_id = '1499d578-978b-4bf5-94a2-a81af51ec203'

    def test_cant_place_ship_after_all_ships_placed(battleship_game, initialized_game_id):
        game = battleship_game.get_game(initialized_game_id)
        additional_ship = ShipPlacement(
            ship_type="carrier", start={"row": 2, "column": "E"}, direction="horizontal"
        )
    
>       with pytest.raises(
            ValueError, match="All ships are already placed. Cannot place more ships."
        ):
E       Failed: DID NOT RAISE <class 'ValueError'>

test_negative.py:58: Failed
```

This output is parsed down to:
```
Test method `test_cant_place_ship_after_all_ships_placed` in `test_negative.py` failed on line 58. 
>       with pytest.raises(
            ValueError, match="All ships are already placed. Cannot place more ships."
        ):
E       Failed: DID NOT RAISE <class 'ValueError'>
```


Since the test files have already been included in the context, it's enough to show exactly what went
wrong.


### Fix code
If a test fails, this will initiate the "fix_code" ability, which includes the test output in the
context and asks GPT to fix the errors or write code if it's missing.

To reduce completion tokens and latency, GPT is instructed to return only the modified code.

The agent then attempts to merge this code with the original code.

An example of the code returned from GPT might look like:
```python
class Battleship(AbstractBattleship):
    # ... rest of the code ... 
    
    def create_ship_placement(self, game_id: str, placement: ShipPlacement) -> None:
        game = self.get_game(game_id)
        if not game:
            raise ValueError("Game not found")
    
        if len(game.ships) == 5:
            raise ValueError("All ships are already placed. Cannot place more ships.")
        
        # ... rest of the code

    # ... rest of the code ... 
```

This code is parsed into "code blocks" and compared to code blocks from the original file.

Getting this to work correctly is a bit of trial and error since GPT tends to return syntactically 
incorrect code when not returning all of the code. 


### Trim code
To reduce context, the agent can trim certain lower-priority files so only class and function
definitions are sent. The format of the code tries to mimic the format GPT uses when commenting
out parts of the code.


### Write test code
In the first iteration, I allowed the agent to also write test cases for the code to be generated.
To optimize this, it was done in two parallel calls, one where the code was written and one where the tests were written.
Then the test was run, and if there were any errors, the "fix code" step was executed.

However, after I added to the write code prompt that GPT should write its interpretation of the 
requirements before writing the code, the results became so robust that I skipped writing the test.
Mostly because the benchmark often timed out. I still believe that the idea is a good way to generate 
new code that is executable.


### Parallel coder
This was an experiment to allow multiple agents to write code simultaneously and choose the code 
where the tests passed. This was primarily to use GPT-3.5 to generate the code. However, it didn't 
fully succeed and became too expensive when using GPT-4.


### Some tradeoffs for speed
To generate a complete solution within the benchmark timeout, some trade-offs were made.

In the first step, the agent only returns the ability to be executed and not its reasoning 
behind it. This is because it simply took too long to generate this text, and it didn't provide 
any better results since the step to choose the ability is relatively small in this agent where
the focus is on managing code.

Instead of letting GPT choose the next step, this is done automatically based on the previous
ability and whether tests have passed.
