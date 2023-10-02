import requests

BASE_URL = 'http://127.0.0.1:8080/ap/v1/'

data1 = {
    "input": "Write the word 'Washington' to a .txt file",
    "eval_id": "f219f3d3-a41b-45a9-a3d0-389832086ee8"
}

data = {
    "input": "Create a three_sum function in a file called sample_code.py. Given an array of integers, return indices of the three numbers such that they add up to a specific target. You may assume that each input would have exactly one solution, and you may not use the same element twice. Example: Given nums = [2, 7, 11, 15], target = 20, Because nums[0] + nums[1] + nums[2] = 2 + 7 + 11 = 20, return [0, 1, 2].",
    "eval_id": "a1ff38a4-1032-4bf2-960a-3b927f9936f4"
}

data2 = {
    "input": "Read the file called file_to_read.txt and write its content to a file called output.txt",
    "eval_id": "f219f3d3-a41b-45a9-a3d0-389832086ee8"
}

data3 = {
    "input": "Create a random password generator. The password should have between 8 and 16 characters and should contain letters, numbers and symbols. "
             "The password should be printed to the console."
             "The entry point will be a python file that can be run this way: python password_generator.py [--len x] "
             "where x is the length of the password. If no length is specified, the password should be 8 characters long. "
             "The password_generator can also be imported as a module and called as password = password_generator.generate_password(len=x). "
             "Any invalid input should raise a ValueError.",
    "eval_id": "ac75c471-e0ce-400c-ba9a-fb72aaab444f"
}


input = """"Build a battleship game

Specifications:

Overview: Battleship is a two-player strategy game where each player places their fleet of ships on a grid and tries to sink the opponent's fleet by guessing their locations.
Players take turns calling out a row and column, attempting to name a square containing one of the opponent's ships.

The Grid: Each player's grid is a 10x10 grid, identified by rows (using numbers 1-10) and columns (using letters A-J).

Ships:

Carrier - 5 squares
Battleship - 4 squares
Cruiser - 3 squares
Submarine - 3 squares
Destroyer - 2 squares
Each ship occupies contiguous squares on the grid, arranged either horizontally or vertically.

Setup:

At the start of the game, each player places their fleet on their grid. This setup is hidden from the opponent.
The game begins with Player 1, followed by Player 2, and so on.
Taking Turns:

On a player's turn, they announce a grid square (e.g., \"D5\").
The opponent announces whether that square is a \"hit\" (if there's a part of a ship on that square) or \"miss\" (if the square is empty).
If a player hits a square occupied by a ship, they get another turn to guess. This continues until they make a miss, at which point their turn ends.
If a player hits all the squares occupied by a ship, the opponent must announce the sinking of that specific ship, e.g., \"You sank my Battleship!\"

Objective: The goal is to sink all of your opponent's ships before they sink yours.

End of the Game: The game ends when one player has sunk all of the opponent's ships. The winner is the player who sinks all the opposing fleet first.

Technical details:
In your root folder you will find an abstract class that defines the public interface of the Battleship class you will have to build:
```
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, validator


# Models for the request and response payloads
class ShipPlacement(BaseModel):
    ship_type: str
    start: dict  # {\"row\": int, \"column\": str}
    direction: str

    @validator(\"start\")
    def validate_start(cls, start):
        row, column = start.get(\"row\"), start.get(\"column\")

        if not (1 <= row <= 10):
            raise ValueError(\"Row must be between 1 and 10 inclusive.\")

        if column not in list(\"ABCDEFGHIJ\"):
            raise ValueError(\"Column must be one of A, B, C, D, E, F, G, H, I, J.\")

        return start


class Turn(BaseModel):
    target: dict  # {\"row\": int, \"column\": str}


class TurnResponse(BaseModel):
    result: str
    ship_type: Optional[str]  # This would be None if the result is a miss


class GameStatus(BaseModel):
    is_game_over: bool
    winner: Optional[str]


from typing import List


class Game(BaseModel):
    game_id: str
    players: List[str]
    board: dict  # This could represent the state of the game board, you might need to flesh this out further
    ships: List[ShipPlacement]  # List of ship placements for this game
    turns: List[Turn]  # List of turns that have been taken


class AbstractBattleship(ABC):
    SHIP_LENGTHS = {
            \"carrier\": 5,
        \"battleship\": 4,
        \"cruiser\": 3,
        \"submarine\": 3,
        \"destroyer\": 2,
    }

    @abstractmethod
    def create_ship_placement(self, game_id: str, placement: ShipPlacement) -> None:
        \"\"\"
        Place a ship on the grid.
        \"\"\"
        pass

    @abstractmethod
    def create_turn(self, game_id: str, turn: Turn) -> TurnResponse:
        \"\"\"
        Players take turns to target a grid cell.
        \"\"\"
        pass

    @abstractmethod
    def get_game_status(self, game_id: str) -> GameStatus:
        \"\"\"
        Check if the game is over and get the winner if there's one.
        \"\"\"
        pass

    @abstractmethod
    def get_winner(self, game_id: str) -> str:
        \"\"\"
        Get the winner of the game.
        \"\"\"
        pass

    @abstractmethod
    def get_game(self) -> Game:
        \"\"\"
        Retrieve the state of the game.
        \"\"\"
        pass

    @abstractmethod
    def delete_game(self, game_id: str) -> None:
        \"\"\"
        Delete a game given its ID.
        \"\"\"
        pass

    @abstractmethod
    def create_game(self) -> None:
        \"\"\"
        Create a new game.
        \"\"\"
        pass

```
At any moment you can run ```pytest``` to execute the tests.
You have two types of test: 
- positive tests => test the battleship game being used in ideal conditions
- negative tests => tests the battleship game behaviour when used incorrectly

Success criteria:
- you will need to write a file called battleship.py that implements the abstract Battleship class.
- this class will have to pass all the tests.
- you're not allowed to modify any other file than the battleship.py. You can add other files as long as the main entrypoint is the battleship class."""


battleship = {
    "input": "input",
    "eval_id": "4d613d05-475f-4f72-bf12-f6d3714340c1"
}

# Step 1: POST request to /agent/tasks
response = requests.post(BASE_URL + 'agent/tasks', json=battleship)
if response.status_code != 200:
    print("Failed to make the request to /agent/tasks")
    exit()

# Extract task_id from the response
task_id = response.json().get("task_id")
if not task_id:
    print("Task ID not found in the response")
    exit()

# Step 2: POST request to /agent/tasks/{task_id}/steps
finished = False
steps_endpoint = f'agent/tasks/{task_id}/steps'
while not finished:
    response_steps = requests.post(BASE_URL + steps_endpoint, json={})
    if response_steps.status_code != 200:
        print(f"Failed to make the request to {steps_endpoint}")
        exit()

    step = response_steps.json()
    print(step)
    if step["is_last"]:
        finished = True

# Step 3: POST request to /agent/tasks/{task_id}/evaluations
eval_endpoint = f'agent/tasks/{task_id}/evaluations'
response_eval = requests.post(BASE_URL + eval_endpoint)
if response_eval.status_code != 200:
    print(f"Failed to make the request to {eval_endpoint}")
    exit()

print("All requests completed successfully!")