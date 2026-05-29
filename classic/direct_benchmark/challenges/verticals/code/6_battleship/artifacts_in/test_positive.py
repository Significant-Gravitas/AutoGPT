from .abstract_class import ShipPlacement, Turn


def test_turns_and_results(battleship_game, initialized_game_id):
    turn = Turn(target={"row": 1, "column": "A"})
    response = battleship_game.create_turn(initialized_game_id, turn)

    assert response.result in ["hit", "miss"]
    if response.result == "hit":
        assert response.ship_type == "carrier"
    game = battleship_game.get_game(initialized_game_id)
    assert turn in game.turns


def test_game_status_and_winner(battleship_game):
    game_id = battleship_game.create_game()
    status = battleship_game.get_game_status(game_id)
    assert isinstance(status.is_game_over, bool)
    if status.is_game_over:
        winner = battleship_game.get_winner(game_id)
        assert winner is not None


def test_delete_game(battleship_game):
    game_id = battleship_game.create_game()
    battleship_game.delete_game(game_id)
    assert battleship_game.get_game(game_id) is None


def test_ship_rotation(battleship_game):
    game_id = battleship_game.create_game()
    placement_horizontal = ShipPlacement(
        ship_type="battleship", start={"row": 1, "column": "B"}, direction="horizontal"
    )
    battleship_game.create_ship_placement(game_id, placement_horizontal)
    placement_vertical = ShipPlacement(
        ship_type="cruiser", start={"row": 3, "column": "D"}, direction="vertical"
    )
    battleship_game.create_ship_placement(game_id, placement_vertical)
    game = battleship_game.get_game(game_id)
    assert placement_horizontal in game.ships
    assert placement_vertical in game.ships


def test_game_state_updates(battleship_game, initialized_game_id):
    turn = Turn(target={"row": 3, "column": "A"})
    battleship_game.create_turn(initialized_game_id, turn)

    game = battleship_game.get_game(initialized_game_id)

    target_key = (3, ord("A") - ord("A"))
    assert target_key in game.board and game.board[target_key] == "hit"


def test_ship_sinking_feedback(battleship_game, initialized_game_id):
    hits = ["A", "B", "C", "D"]
    static_moves = [
        {"row": 1, "column": "E"},
        {"row": 1, "column": "F"},
        {"row": 1, "column": "G"},
        {"row": 1, "column": "H"},
    ]

    response = None
    for index, hit in enumerate(hits):
        turn = Turn(target={"row": 2, "column": hit})
        response = battleship_game.create_turn(initialized_game_id, turn)
        assert response.ship_type == "battleship"

        static_turn = Turn(target=static_moves[index])
        battleship_game.create_turn(initialized_game_id, static_turn)

    assert response and response.result == "sunk"


def test_restart_game(battleship_game):
    game_id = battleship_game.create_game()
    battleship_game.delete_game(game_id)
    game_id = (
        battleship_game.create_game()
    )  # Use the returned game_id after recreating the game
    game = battleship_game.get_game(game_id)
    assert game is not None


def test_ship_edge_overlapping(battleship_game):
    game_id = battleship_game.create_game()

    first_ship = ShipPlacement(
        ship_type="battleship", start={"row": 1, "column": "A"}, direction="horizontal"
    )
    battleship_game.create_ship_placement(game_id, first_ship)

    next_ship = ShipPlacement(
        ship_type="cruiser", start={"row": 1, "column": "E"}, direction="horizontal"
    )
    battleship_game.create_ship_placement(game_id, next_ship)

    game = battleship_game.get_game(game_id)
    assert first_ship in game.ships
    assert next_ship in game.ships


def test_game_state_after_ship_placement(battleship_game):
    game_id = battleship_game.create_game()

    ship_placement = ShipPlacement(
        ship_type="battleship", start={"row": 1, "column": "A"}, direction="horizontal"
    )
    battleship_game.create_ship_placement(game_id, ship_placement)

    game = battleship_game.get_game(game_id)
    assert ship_placement in game.ships


def test_game_state_after_turn(initialized_game_id, battleship_game):
    turn = Turn(target={"row": 1, "column": "A"})
    response = battleship_game.create_turn(initialized_game_id, turn)

    game = battleship_game.get_game(initialized_game_id)

    if response.result == "hit":
        assert game.board[(1, 0)] == "hit"
    else:
        assert game.board[1][0] == "miss"


def test_multiple_hits_on_ship(battleship_game, initialized_game_id):
    hit_positions = ["A", "B", "C", "D", "E"]

    for index, pos in enumerate(hit_positions):
        turn = Turn(target={"row": 1, "column": pos})
        response = battleship_game.create_turn(initialized_game_id, turn)

        if index == len(hit_positions) - 1:
            assert response.result == "sunk"
        else:
            assert response.result == "hit"


def test_game_over_condition(battleship_game, initialized_game_id):
    for row in range(1, 11):
        for column in list("ABCDEFGHIJ"):
            turn = Turn(target={"row": row, "column": column})
            battleship_game.create_turn(initialized_game_id, turn)

            battleship_game.create_turn(initialized_game_id, turn)

    status = battleship_game.get_game_status(initialized_game_id)
    assert status.is_game_over
