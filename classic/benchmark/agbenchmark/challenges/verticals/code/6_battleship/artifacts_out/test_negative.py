import pytest
from pydantic import ValidationError

from .abstract_class import ShipPlacement, Turn


def test_ship_placement_out_of_bounds(battleship_game):
    game_id = battleship_game.create_game()

    try:
        out_of_bounds_ship = ShipPlacement(
            ship_type="battleship",
            start={"row": 11, "column": "Z"},
            direction="horizontal",
        )
    except ValidationError:  # Use the directly imported ValidationError class
        pass
    else:
        with pytest.raises(ValueError, match="Placement out of bounds"):
            battleship_game.create_ship_placement(game_id, out_of_bounds_ship)


def test_no_ship_overlap(battleship_game):
    game_id = battleship_game.create_game()
    placement1 = ShipPlacement(
        ship_type="battleship", start={"row": 1, "column": "A"}, direction="horizontal"
    )
    battleship_game.create_ship_placement(game_id, placement1)
    placement2 = ShipPlacement(
        ship_type="cruiser", start={"row": 1, "column": "A"}, direction="horizontal"
    )
    with pytest.raises(ValueError):
        battleship_game.create_ship_placement(game_id, placement2)


def test_cant_hit_before_ships_placed(battleship_game):
    game_id = battleship_game.create_game()
    placement1 = ShipPlacement(
        ship_type="battleship", start={"row": 1, "column": "A"}, direction="horizontal"
    )
    battleship_game.create_ship_placement(game_id, placement1)
    placement2 = ShipPlacement(
        ship_type="cruiser", start={"row": 4, "column": "D"}, direction="horizontal"
    )
    battleship_game.create_ship_placement(game_id, placement2)
    turn = Turn(target={"row": 1, "column": "A"})
    with pytest.raises(
        ValueError, match="All ships must be placed before starting turns"
    ):
        battleship_game.create_turn(game_id, turn)


def test_cant_place_ship_after_all_ships_placed(battleship_game, initialized_game_id):
    battleship_game.get_game(initialized_game_id)
    additional_ship = ShipPlacement(
        ship_type="carrier", start={"row": 2, "column": "E"}, direction="horizontal"
    )

    with pytest.raises(
        ValueError, match="All ships are already placed. Cannot place more ships."
    ):
        battleship_game.create_ship_placement(initialized_game_id, additional_ship)


def test_ship_placement_invalid_direction(battleship_game):
    game_id = battleship_game.create_game()

    with pytest.raises(ValueError, match="Invalid ship direction"):
        invalid_direction_ship = ShipPlacement(
            ship_type="battleship",
            start={"row": 1, "column": "A"},
            direction="diagonal",
        )
        battleship_game.create_ship_placement(game_id, invalid_direction_ship)


def test_invalid_ship_type(battleship_game):
    game_id = battleship_game.create_game()
    invalid_ship = ShipPlacement(
        ship_type="spacecraft", start={"row": 1, "column": "A"}, direction="horizontal"
    )
    with pytest.raises(ValueError, match="Invalid ship type"):
        battleship_game.create_ship_placement(game_id, invalid_ship)


def test_ship_placement_extends_beyond_boundaries(battleship_game):
    game_id = battleship_game.create_game()

    with pytest.raises(ValueError, match="Ship extends beyond board boundaries"):
        ship_extending_beyond = ShipPlacement(
            ship_type="battleship",
            start={"row": 1, "column": "H"},
            direction="horizontal",
        )
        battleship_game.create_ship_placement(game_id, ship_extending_beyond)

    with pytest.raises(ValueError, match="Ship extends beyond board boundaries"):
        ship_extending_beyond = ShipPlacement(
            ship_type="cruiser", start={"row": 9, "column": "A"}, direction="vertical"
        )
        battleship_game.create_ship_placement(game_id, ship_extending_beyond)
