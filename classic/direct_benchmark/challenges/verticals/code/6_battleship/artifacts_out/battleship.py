from typing import Dict

from .abstract_class import (
    AbstractBattleship,
    Game,
    GameStatus,
    ShipPlacement,
    Turn,
    TurnResponse,
)


class Battleship(AbstractBattleship):
    def __init__(self):
        self.games: Dict[str, Game] = {}

    def create_game(self) -> str:
        game_id = str(len(self.games))
        new_game = Game(
            game_id=game_id,
            players=[],
            board={},
            ships=[],
            turns=[],
        )

        self.games[game_id] = new_game
        return game_id

    def create_ship_placement(self, game_id: str, placement: ShipPlacement) -> None:
        game = self.games.get(game_id)

        if not game:
            raise ValueError(f"Game with ID {game_id} not found.")
        if placement.direction not in ["horizontal", "vertical"]:
            raise ValueError("Invalid ship direction")
        if self.all_ships_placed(game):
            raise ValueError("All ships are already placed. Cannot place more ships.")

        ship_length = self.SHIP_LENGTHS.get(placement.ship_type)
        if not ship_length:
            raise ValueError(f"Invalid ship type {placement.ship_type}")

        start_row, start_col = placement.start["row"], ord(
            placement.start["column"]
        ) - ord("A")

        if start_row < 1 or start_row > 10 or start_col < 0 or start_col > 9:
            raise ValueError("Placement out of bounds")

        if placement.direction == "horizontal" and start_col + ship_length > 10:
            raise ValueError("Ship extends beyond board boundaries")
        elif placement.direction == "vertical" and start_row + ship_length > 10:
            raise ValueError("Ship extends beyond board boundaries")

        for i in range(ship_length):
            if placement.direction == "horizontal":
                if game.board.get((start_row, start_col + i)):
                    raise ValueError("Ship overlaps with another ship!")
            elif placement.direction == "vertical":
                if game.board.get((start_row + i, start_col)):
                    raise ValueError("Ship overlaps with another ship!")

        for i in range(ship_length):
            if placement.direction == "horizontal":
                game.board[(start_row, start_col + i)] = placement.ship_type
            else:
                game.board[(start_row + i, start_col)] = placement.ship_type

        game.ships.append(placement)

    def create_turn(self, game_id: str, turn: Turn) -> TurnResponse:
        game = self.games.get(game_id)

        if not game:
            raise ValueError(f"Game with ID {game_id} not found.")

        if not self.all_ships_placed(game):
            raise ValueError("All ships must be placed before starting turns")

        target_row, target_col = turn.target["row"], ord(turn.target["column"]) - ord(
            "A"
        )
        hit_ship = game.board.get((target_row, target_col))

        game.turns.append(turn)

        if not hit_ship or hit_ship == "hit":  # if no ship or already hit
            return TurnResponse(result="miss", ship_type=None)

        ship_placement = next(sp for sp in game.ships if sp.ship_type == hit_ship)
        start_row, start_col = (
            ship_placement.start["row"],
            ord(ship_placement.start["column"]) - ord("A"),
        )
        ship_positions = [
            (
                start_row + (i if ship_placement.direction == "vertical" else 0),
                start_col + (i if ship_placement.direction == "horizontal" else 0),
            )
            for i in range(self.SHIP_LENGTHS[hit_ship])
        ]

        targeted_positions = {
            (t.target["row"], ord(t.target["column"]) - ord("A")) for t in game.turns
        }

        game.board[(target_row, target_col)] = "hit"

        if set(ship_positions).issubset(targeted_positions):
            for pos in ship_positions:
                game.board[pos] = "hit"
            return TurnResponse(result="sunk", ship_type=hit_ship)
        else:
            return TurnResponse(result="hit", ship_type=hit_ship)

    def get_game_status(self, game_id: str) -> GameStatus:
        game = self.games.get(game_id)

        if not game:
            raise ValueError(f"Game with ID {game_id} not found.")

        hits = sum(1 for _, status in game.board.items() if status == "hit")

        total_ships_length = sum(
            self.SHIP_LENGTHS[ship.ship_type] for ship in game.ships
        )

        if hits == total_ships_length:
            return GameStatus(is_game_over=True, winner="player")
        else:
            return GameStatus(is_game_over=False, winner=None)

    def get_winner(self, game_id: str) -> str:
        game_status = self.get_game_status(game_id)

        if game_status.is_game_over and game_status.winner:
            return game_status.winner
        else:
            raise ValueError(f"Game {game_id} isn't over yet")

    def get_game(self, game_id: str) -> Game | None:
        return self.games.get(game_id)

    def delete_game(self, game_id: str) -> None:
        if game_id in self.games:
            del self.games[game_id]

    def all_ships_placed(self, game: Game) -> bool:
        placed_ship_types = set([placement.ship_type for placement in game.ships])
        return placed_ship_types == set(self.SHIP_LENGTHS.keys())
