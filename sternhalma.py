from dataclasses import dataclass
from enum import IntEnum
from types import FunctionType
from typing import final, override

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Player(IntEnum):
    Player1 = 1
    Player2 = 2

    @classmethod
    def from_str(cls, player_str: str) -> "Player":
        match player_str:
            case "1":
                return Player.Player1
            case "2":
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
# Array of shape (2,)
type BoardIndex = NDArray[np.int_]


# How many steps it takes to get from one cell to another
def hexagonal_metric(d: BoardIndex) -> np.int_:
    return np.max(np.abs([d[0], d[1], d[0] + d[1]]))


def hexagonal_distance(i: BoardIndex, j: BoardIndex) -> np.int_:
    return hexagonal_metric(j - i)


# Euclidean metric on the hexagonal grid
def euclidean_metric(d: BoardIndex) -> np.float64:
    return np.sqrt(np.square(d[0] + d[1]) - (d[0] * d[1]))


def euclidean_distance(i: BoardIndex, j: BoardIndex) -> np.float64:
    return euclidean_metric(j - i)


# Valid positions on the board
# Stored as tuple of
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

# Starting positions of player 1
PLAYER1_STARTING_POSITIONS: tuple[NDArray[np.int_], NDArray[np.int_]] = tuple(np.transpose([
[12,4],[12,5],[12,6],[12,7],[12,8],
    [13,4],[13,5],[13,6],[13,7],
       [14,4],[14,5],[14,6],
           [15,4],[15,5],
              [16,4],
]))  # fmt: skip

# Starting positions of player 2
PLAYER2_STARTING_POSITIONS: tuple[NDArray[np.int_], NDArray[np.int_]] = tuple(np.transpose([
            [0,12],
        [1,11],[1,12],
     [2,10],[2,11],[2,12],
  [3,9],[3,10],[3,11],[3,12],
[4,8],[4,9],[4,10],[4,11],[4,12],
]))  # fmt: skip


class Position(IntEnum):
    Invalid = -1
    Empty = 0
    Player1 = 1
    Player2 = 2

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


# Movement of a piece on the board represented by a pair of board indices
# Array of shape (2, 2)
type Movement = NDArray[np.int_]


@final
@dataclass
class Board:
    state: NDArray[np.int32]

    @classmethod
    def empty(cls) -> "Board":
        state = np.full((17, 17), Position.Invalid, dtype=np.int32)
        state[VALID_POSITIONS] = Position.Empty
        return cls(state=state)

    @classmethod
    def two_players(cls) -> "Board":
        board = cls.empty()
        board.state[PLAYER1_STARTING_POSITIONS] = Position.Player1
        board.state[PLAYER2_STARTING_POSITIONS] = Position.Player2
        return board

    def __getitem__(self, idx: BoardIndex) -> Position:
        return Position(self.state[tuple(idx)])

    def __setitem__(self, idx: BoardIndex, position: Position) -> None:
        self.state[tuple(idx)] = position

    def apply_movement(self, movement: Movement) -> None:
        self[movement[1]] = self[movement[0]]
        self[movement[0]] = Position.Empty

    def print(self):
        print(
            "\n".join(
                map(
                    lambda e: " " * e[0]
                    + "".join(map(lambda x: str(Position(x)), e[1])),
                    enumerate(self.state),
                )
            )
        )


@dataclass(frozen=True)
class GameResult:
    winner: Player
