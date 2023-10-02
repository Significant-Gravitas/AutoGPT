import asyncio

import forge
from forge.sdk import LocalWorkspace, TaskRequestBody, StepRequestBody
import forge.agent
import forge.sdk.db

database_name = "sqlite:///agent.db"
workspace = LocalWorkspace("/home/albert/repos/albert/AutoGPT/autogpts/ghostcoder/workspace")

database = forge.sdk.db.AgentDB(database_name, debug_enabled=False)
agent = forge.agent.ForgeAgent(database=database, workspace=workspace)

async def execute(input: str):
    task = await agent.create_task(TaskRequestBody(input=input))
    print(task.task_id)
    finished = False
    while not finished:
        step = await agent.execute_step(task.task_id, StepRequestBody(input=input))
        if step.is_last:
            finished = True

async def execute_task(task_id):
    finished = False
    while not finished:
        step = await agent.execute_step(task_id, StepRequestBody(input=input))
        if step.is_last:
            finished = True



input = "Create a three_sum function in a file called sample_code.py. Given an array of integers, return indices of the three numbers such that they add up to a specific target. You may assume that each input would have exactly one solution, and you may not use the same element twice. Example: Given nums = [2, 7, 11, 15], target = 20, Because nums[0] + nums[1] + nums[2] = 2 + 7 + 11 = 20, return [0, 1, 2]."
input = "Read the file called file_to_read.txt and write its content to a file called output.txt"

input = """Create a random password generator. The password should have between 8 and 16 characters and should contain letters, numbers and symbols. 
The password should be printed to the console.
The entry point will be a python file that can be run this way: python password_generator.py [--len x] 
where x is the length of the password. If no length is specified, the password should be 8 characters long. 
The password_generator can also be imported as a module and called as password = password_generator.generate_password(len=x). 
Any invalid input should raise a ValueError."""

input = """Create a file organizer CLI tool in Python that sorts files in a directory based on their file types (e.g., 
images, documents, audio) and moves them into these corresponding folders: 'images', 'documents', 'audio'.
The entry point will be a python file that can be run this way: python organize_files.py --directory_path=YOUR_DIRECTORY_PATH"
"""

input = """Build a basic URL shortener using a python CLI. Here are the specifications.

Functionality: 
The program should have two primary functionalities.

horten a given URL.
Retrieve the original URL from a shortened URL.

CLI: The command-line interface should accept a URL as its first input. 
It should be able to determine if the url is a shortened url or not. If the url is not shortened, 
it will display ONLY the shortened url, otherwise, it will display ONLY the original unshortened URL.
 Afterwards, it should prompt the user for another URL to process.
 
Technical specifications:
Build a file called url_shortener.py. This file will be called through command lines.

Edge cases:
For the sake of simplicity, there will be no edge cases, you can assume the input is always correct and the user
immediately passes the shortened version of the url he just shortened.
You will be expected to create a python file called url_shortener.py that will run through command lines by using 
python url_shortener.py.
 
The url_shortener.py will be tested this way:
```
import unittest
from url_shortener import shorten_url, retrieve_url
class TestURLShortener(unittest.TestCase):
    def test_url_retrieval(self):
    # Shorten the URL to get its shortened form
    shortened_url = shorten_url('https://www.example.com')

    # Retrieve the original URL using the shortened URL directly
    retrieved_url = retrieve_url(shortened_url)
    
    self.assertEqual(retrieved_url, 'https://www.example.com', \"Retrieved URL does not match the original!\")
    
if __name__ == \"__main__\":
    unittest.main()
```"""

input = """Build a Tic-Tac-Toe game using a python CLI. Here are the specifications.

The Grid: The game board is a 3x3 grid, consisting of 3 rows and 3 columns, creating a total of 9 squares.

Players: There are two players. One player uses the number \"1\", and the other player uses the number \"2\".

Taking Turns: Players take turns to put their respective numbers (\"1\" or \"2\") in an empty square of the grid. Once a player has placed their number in a square, it cannot be changed or removed.

Objective: The goal is to get three of your numbers in a row, either horizontally, vertically, or diagonally.

End of the Game: The game concludes in one of two ways: One player gets three of their numbers in a row (horizontally, vertically, or diagonally) and is declared the winner.
All squares on the grid are filled, and no player has three in a row. This situation is a \"draw\" or a \"tie\".

Technical specifications:
Build a file called tic_tac_toe.py. This file will be called through command lines. You will have to prompt users for their move. Player 1 will always start.
Players will input their move in the following format: \"x,y\" where x and y represent the location in the grid (0,0 is top left, 2,2 is bottom right).

Your primary requirement is to halt the game when appropriate and to print only one of these three exact sentences:

\"Player 1 won!\"
\"Player 2 won!\"
\"Draw\"

Edge cases: A player can send an incorrect location. Either the location is incorrect or the square is already filled. In this case, this counts as doing nothing, and the player gets prompted for new locations again.


You will be expected to create a python file called tic_tac_toe.py that will run through command lines by using ```python tic_tac_toe.py```.

Here is an example of how your tic_tac_toe.py game will be tested.
```
process = subprocess.Popen(
        ['python', 'tic_tac_toe.py'],
    stdout=subprocess.PIPE,
    text=True
)

output, _ = process.communicate('\
'.join([\"0,0\", \"1,0\", \"0,1\", \"1,1\", \"0,2\"]))

assert \"Player 1 won!\" in output
```"""

input = """Build a battleship game

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
    def create_game(self, game_id: str) -> None:
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

#output = asyncio.run(create_task(input))
output = asyncio.run(execute_task("c81e40e7-fb83-46e0-88be-74cdb321396a"))

