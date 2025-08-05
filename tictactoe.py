"""Tic-Tac-Toe Game Implementation

This module implements the game logic and an agent capable of playing the game.
"""

from dataclasses import dataclass
import random
from enum import IntEnum
from typing import Self, final, override


class Player(IntEnum):
    """Represents a player in the Tic-Tac-Toe game.

    Attributes:
        Empty: Unoccupied cell.
        Nought: Nought player (usually "o").
        Cross: Cross player (usually "x").
    """

    Empty = 0
    Nought = 1
    Cross = 2

    @override
    def __str__(self):
        match self:
            case Player.Empty:
                return "⬜"
            case Player.Nought:
                return "⭕"
            case Player.Cross:
                return "❌"

    @classmethod
    def from_str(cls, player_str: str | None):
        """Converts a string representation of a player to a Player enum.

        Args:
            player_str: String representation of the player, can be "o", "x", or None.

        Returns:
            Player: Corresponding Player enum value.

        Raises:
            ValueError: If the string does not match any known player representation.
        """

        match player_str:
            case "o":
                return Player.Nought
            case "x":
                return Player.Cross
            case None:
                return Player.Empty
            case _:
                raise ValueError(f"Unknown player string: {player_str}")

    def opponent(self):
        """Returns the opponent of the current player.

        Returns:
            Player: The opponent player.

        Raises:
            ValueError: If the current player is Empty, since it has no opponent.
        """

        match self:
            case Player.Nought:
                return Player.Cross
            case Player.Cross:
                return Player.Nought
            case Player.Empty:
                raise ValueError("Empty player has no opponent.")


@final
class Board:
    """Represents the Tic-Tac-Toe game board.

    Attributes:
        board: 2D list representing the game board, where each cell can be Empty, Nought, or Cross.
    """

    def __init__(self, board: list[list[Player]]):
        """Initializes the Tic-Tac-Toe board with a given state.

        Args:
            board: 2D list of Player enums representing the initial state of the board.
        """

        self.board = board

    @classmethod
    def empty(cls) -> Self:
        """Creates an empty Tic-Tac-Toe board.

        Returns:
            Board: An instance of the Board class with all cells empty.
        """

        return cls([[Player.Empty for _ in range(3)] for _ in range(3)])

    @classmethod
    def from_json(cls, board_json: list[list[str]]) -> Self:
        """Creates a Board instance from a JSON representation.

        Args:
            board_json: 2D list of strings representing the board state.

        Returns:
            Board: An instance of the Board class with the specified state.
        """

        return cls(list(map(lambda row: list(map(Player.from_str, row)), board_json)))

    @override
    def __str__(self):
        return "\n".join("".join(f"{cell}" for cell in row) for row in self.board)

    def __setitem__(self, position: tuple[int, int], player: Player):
        row, col = position
        if self.board[row][col] == Player.Empty.value:
            self.board[row][col] = player
        else:
            raise ValueError("Cell is already occupied.")


@dataclass(frozen=True)
class GameResult:
    pass


@final
@dataclass(frozen=True)
class GameResultDraw(GameResult):
    pass


@final
@dataclass(frozen=True)
class GameResultVictory(GameResult):
    player: Player


@final
class Agent:
    """Represents a Tic-Tac-Toe agent that plays the game.

    Attributes:
        player: The player type (Nought or Cross) that the agent will play as.
        board: The current state of the game board.
    """

    def __init__(self, player: Player):
        self.player = player
        self.board = Board.empty()

    def decide_opening(self) -> tuple[int, int]:
        chosen_opening = random.choice([(i, j) for i in range(3) for j in range(3)])
        return chosen_opening

    def decide_move(self, available_moves: list[tuple[int, int]]) -> tuple[int, int]:
        chosen_move = random.choice(available_moves)
        return chosen_move

    def make_move(self, position: tuple[int, int]):
        self.board[position] = self.player
