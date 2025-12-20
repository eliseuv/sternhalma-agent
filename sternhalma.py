"""Sternhalma game module containing board, player, and logic definitions.

This module defines the core data structures and logic for the Sternhalma (Chinese Checkers) game.
It includes definitions for the board, players, valid positions, and metrics for calculating distances
on the hexagonal grid.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import final, override

import numpy as np
from numpy.typing import NDArray


class Player(IntEnum):
    """
    Represents the two players in the game.

    Player 1 is the starting player (Blue, 🔵).
    Player 2 is the second player (Red, 🔴).
    """

    Player1 = 1
    Player2 = 2

    @classmethod
    def from_str(cls, player_str: str) -> "Player":
        match player_str:
            case "player1":
                return Player.Player1
            case "player2":
                return Player.Player2
            case _:
                raise ValueError(f"Unknown player string: {player_str}")

    @override
    def __str__(self) -> str:
        match self:
            case Player.Player1:
                return "🔵"
            case Player.Player2:
                return "🔴"


# Axial index in the hexagonal board
# The board uses a 2D coordinate system (q, r) where q is the column and r is the row (or derived axis).
# Array of shape (2,) representing [q, r]
# See https://www.redblobgames.com/grids/hexagons/ for hexagonal grid coordinate systems.
type BoardIndex = NDArray[np.int_]


# How many steps it takes to get from one cell to another
def hexagonal_metric(d: BoardIndex) -> np.int_:
    """
    Calculates the hexagonal distance metric (Manhattan distance on hex grid) from the origin.

    Args:
        d: The difference vector in axial coordinates.

    Returns:
        The number of steps to reach the destination from the origin.
    """
    return np.max(np.abs([d[0], d[1], d[0] + d[1]]))


def hexagonal_distance(i: BoardIndex, j: BoardIndex) -> np.int_:
    """
    Calculates the number of steps (distance) between two hexagonal coordinates.

    Args:
        i: The starting coordinate.
        j: The ending coordinate.

    Returns:
        The integer distance between i and j.
    """
    return hexagonal_metric(j - i)


# Euclidean metric on the hexagonal grid
def euclidean_metric(d: BoardIndex) -> np.float64:
    """
    Calculates the Euclidean distance from the origin in the hexagonal grid embedding.

    This accounts for the geometry of the hexagons where axes are at 60 degrees.
    Based on the formula: distance = sqrt(x^2 + y^2 + xy) for axial coordinates (if axes are 120 deg apart).
    Here implemented as sqrt((q+r)^2 - qr) which is equivalent to standard Euclidean distance on 60-degree basis.

    Args:
        d: The difference vector.

    Returns:
        The Euclidean distance.
    """
    return np.sqrt(np.square(d[0] + d[1]) - (d[0] * d[1]))


def euclidean_distance(i: BoardIndex, j: BoardIndex) -> np.float64:
    """
    Calculates the Euclidean distance between two coordinates.

    Args:
        i: The starting coordinate.
        j: The ending coordinate.

    Returns:
        The Euclidean distance.
    """
    return euclidean_metric(j - i)


# Valid positions on the board
# Defined as hardcoded axial coordinates for the standard Sternhalma star-shaped board.
# Stored as tuple of 2 arrays (q_indices, r_indices) for easy indexing.
VALID_POSITIONS: tuple[NDArray[np.int_], NDArray[np.int_]] = tuple(np.transpose([
                                           [0,12],
                                       [1,11],[1,12],
                                    [2,10],[2,11],[2,12],
                                 [3,9],[3,10],[3,11],[3,12],
       [4,4],[4,5],[4,6],[4,7],[4,8],[4,9],[4,10],[4,11],[4,12],[4,13],[4,14],[4,15],[4,16],
          [5,4],[5,5],[5,6],[5,7],[5,8],[5,9],[5,10],[5,11],[5,12],[5,13],[5,14],[5,15],
             [6,4],[6,5],[6,6],[6,7],[6,8],[6,9],[6,10],[6,11],[6,12],[6,13],[6,14],
                [7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],[7,11],[7,12],[7,13],
                   [8,4],[8,5],[8,6],[8,7],[8,8],[8,9],[8,10],[8,11],[8,12],
                [9,3],[9,4],[9,5],[9,6],[9,7],[9,8],[9,9],[9,10],[9,11],[9,12],
       [10,2],[10,3],[10,4],[10,5],[10,6],[10,7],[10,8],[10,9],[10,10],[10,11],[10,12],
    [11,1],[11,2],[11,3],[11,4],[11,5],[11,6],[11,7],[11,8],[11,9],[11,10],[11,11],[11,12],
[12,0],[12,1],[12,2],[12,3],[12,4],[12,5],[12,6],[12,7],[12,8],[12,9],[12,10],[12,11],[12,12],
                                [13,4],[13,5],[13,6],[13,7],
                                   [14,4],[14,5],[14,6],
                                       [15,4],[15,5],
                                          [16,4],
]))  # fmt: skip

# Starting positions of player 1 (Bottom triangle)
# These are the target positions for Player 2.
PLAYER1_STARTING_POSITIONS: tuple[NDArray[np.int_], NDArray[np.int_]] = tuple(np.transpose([
[12,4],[12,5],[12,6],[12,7],[12,8],
    [13,4],[13,5],[13,6],[13,7],
       [14,4],[14,5],[14,6],
           [15,4],[15,5],
              [16,4],
]))  # fmt: skip

# Starting positions of player 2 (Top triangle)
# These are the target positions for Player 1.
PLAYER2_STARTING_POSITIONS: tuple[NDArray[np.int_], NDArray[np.int_]] = tuple(np.transpose([
            [0,12],
        [1,11],[1,12],
     [2,10],[2,11],[2,12],
  [3,9],[3,10],[3,11],[3,12],
[4,8],[4,9],[4,10],[4,11],[4,12],
]))  # fmt: skip

# Board mask to filter out invalid positions
BOARD_MASK = np.zeros((17, 17), dtype=np.float32)
BOARD_MASK[VALID_POSITIONS] = 1


class Position(IntEnum):
    """
    Represents the state of a single cell on the board.
    """

    Invalid = -1  # Off-board cell
    Empty = 0  # Empty valid cell
    Player1 = 1  # Occupied by Player 1
    Player2 = 2  # Occupied by Player 2

    @classmethod
    def with_player(cls, player: Player) -> "Position":
        match player:
            case Player.Player1:
                return Position.Player1
            case Player.Player2:
                return Position.Player2

    @override
    def __str__(self) -> str:
        match self:
            case Position.Invalid:
                return "󠀠󠀠󠀠󠀠   "
            case Position.Empty:
                return "⚫ "
            case Position.Player1:
                return f"{Player.Player1} "
            case Position.Player2:
                return f"{Player.Player2} "


# Movement of a piece on the board represented by a pair of board indices (Start, End)
# Array of shape (2, 2) where [0] is start position and [1] is end position.
# Array of shape (2, 2)
type Movement = NDArray[np.int_]


@final
@dataclass
class Board:
    """
    Represents the game board state.

    The state is a 17x17 grid where valid positions form the star shape.
    Values in the grid are integers mapping to `Position` enum.
    """

    state: NDArray[np.int32]

    @classmethod
    def empty(cls) -> "Board":
        """Creates an empty board with all valid positions set to Empty."""
        state = np.full((17, 17), Position.Invalid, dtype=np.int32)
        state[VALID_POSITIONS] = Position.Empty
        return cls(state=state)

    @classmethod
    def two_players(cls) -> "Board":
        """Creates a standard starting board for a 2-player game."""
        board = cls.empty()
        board.state[PLAYER1_STARTING_POSITIONS] = Position.Player1
        board.state[PLAYER2_STARTING_POSITIONS] = Position.Player2
        return board

    def __getitem__(self, idx: BoardIndex) -> Position:
        return Position(self.state[tuple(idx)])

    def __setitem__(self, idx: BoardIndex, position: Position) -> None:
        self.state[tuple(idx)] = position

    def apply_movement(self, movement: Movement) -> None:
        """
        Applies a movement to the board state in-place.

        Args:
            movement: A (2, 2) array where [0] is start index and [1] is end index.
        """
        self[movement[1]] = self[movement[0]]
        self[movement[0]] = Position.Empty

    def to_string(self) -> str:
        """Returns a string representation of the board for debugging."""
        return "\n".join(
            map(
                lambda e: " " * e[0] + "".join(map(lambda x: str(Position(x)), e[1])),
                enumerate(self.state),
            )
        )


# Scores of each player
type Scores = tuple[int, int]


@dataclass(frozen=True)
class GameResult:
    """
    Data class representing the result of a completed game.
    """

    winner: Player
    total_turns: int
    scores: Scores
