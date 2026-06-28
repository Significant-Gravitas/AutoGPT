import pytest

from .abstract_class import ShipPlacement, Turn
from .battleship import Battleship


@pytest.fixture
def battleship_game():
    return Battleship()


@pytest.fixture
def initialized_game_id(battleship_game):
    # Create a game instance
    game_id = battleship_game.create_game()

    # Place all the ships using battleship_game's methods
    sample_ship_placements = [
        ShipPlacement(
            ship_type="carrier", start={"row": 1, "column": "A"}, direction="horizontal"
        ),
        ShipPlacement(
            ship_type="battleship",
            start={"row": 2, "column": "A"},
            direction="horizontal",
        ),
        ShipPlacement(
            ship_type="cruiser", start={"row": 3, "column": "A"}, direction="horizontal"
        ),
        ShipPlacement(
            ship_type="submarine",
            start={"row": 4, "column": "A"},
            direction="horizontal",
        ),
        ShipPlacement(
            ship_type="destroyer",
            start={"row": 5, "column": "A"},
            direction="horizontal",
        ),
    ]

    for ship_placement in sample_ship_placements:
        # Place ship using battleship_game's methods
        battleship_game.create_ship_placement(game_id, ship_placement)

    return game_id


@pytest.fixture
def game_over_fixture(battleship_game, initialized_game_id):
    # Assuming 10x10 grid, target all possible positions
    for row in range(1, 11):
        for column in list("ABCDEFGHIJ"):
            # Player 1 takes a turn
            turn = Turn(target={"row": row, "column": column})
            battleship_game.create_turn(initialized_game_id, turn)

            # Player 2 takes a turn, targeting the same position as Player 1
            battleship_game.create_turn(initialized_game_id, turn)

    # At the end of this fixture, the game should be over
    return initialized_game_id
