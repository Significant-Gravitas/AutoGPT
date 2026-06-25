from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, field_validator


# Models for the request and response payloads
class ShipPlacement(BaseModel):
    ship_type: str
    start: dict  # {"row": int, "column": str}
    direction: str

    @field_validator("start")
    def validate_start(cls, start):
        row, column = start.get("row"), start.get("column")

        if not (1 <= row <= 10):
            raise ValueError("Row must be between 1 and 10 inclusive.")

        if column not in list("ABCDEFGHIJ"):
            raise ValueError("Column must be one of A, B, C, D, E, F, G, H, I, J.")

        return start


class Turn(BaseModel):
    target: dict  # {"row": int, "column": str}


class TurnResponse(BaseModel):
    result: str
    ship_type: Optional[str]  # This would be None if the result is a miss


class GameStatus(BaseModel):
    is_game_over: bool
    winner: Optional[str]


class Game(BaseModel):
    game_id: str
    players: list[str]
    # This could represent the state of the game board,
    # you might need to flesh this out further:
    board: dict
    ships: list[ShipPlacement]  # List of ship placements for this game
    turns: list[Turn]  # List of turns that have been taken


class AbstractBattleship(ABC):
    SHIP_LENGTHS = {
        "carrier": 5,
        "battleship": 4,
        "cruiser": 3,
        "submarine": 3,
        "destroyer": 2,
    }

    @abstractmethod
    def create_ship_placement(self, game_id: str, placement: ShipPlacement) -> None:
        """
        Place a ship on the grid.
        """
        pass

    @abstractmethod
    def create_turn(self, game_id: str, turn: Turn) -> TurnResponse:
        """
        Players take turns to target a grid cell.
        """
        pass

    @abstractmethod
    def get_game_status(self, game_id: str) -> GameStatus:
        """
        Check if the game is over and get the winner if there's one.
        """
        pass

    @abstractmethod
    def get_winner(self, game_id: str) -> str:
        """
        Get the winner of the game.
        """
        pass

    @abstractmethod
    def get_game(self) -> Game | None:
        """
        Retrieve the state of the game.
        """
        pass

    @abstractmethod
    def delete_game(self, game_id: str) -> None:
        """
        Delete a game given its ID.
        """
        pass

    @abstractmethod
    def create_game(self) -> None:
        """
        Create a new game.

        Returns:
            str: The ID of the created game.
        """
        pass
