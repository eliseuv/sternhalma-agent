from dataclasses import dataclass
from enum import IntEnum
import random
from typing import final, override

import numpy as np
from numpy.typing import NDArray


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

    def __str__(self) -> str:
        match self:
            case Player.Player1:
                return "🔵"
            case Player.Player2:
                return "🔴"


@dataclass(frozen=True)
class GameResult:
    winner: Player


# Axial index in the hexagonal board
type BoardIndex = tuple[int, int]


# Valid positions on the board
VALID_POSITIONS: list[BoardIndex] = [
                                           (0,12),
                                       (1,11),(1,12),
                                    (2,10),(2,11),(2,12),
                                 (3,9),(3,10),(3,11),(3,12),
       (4,4),(4,5),(4,6),(4,7),(4,8),(4,9),(4,10),(4,11),(4,12),(4,13),(4,14),(4,15),(4,16),
          (5,4),(5,5),(5,6),(5,7),(5,8),(5,9),(5,10),(5,11),(5,12),(5,13),(5,14),(5,15),
             (6,4),(6,5),(6,6),(6,7),(6,8),(6,9),(6,10),(6,11),(6,12),(6,13),(6,14),
                (7,4),(7,5),(7,6),(7,7),(7,8),(7,9),(7,10),(7,11),(7,12),(7,13),
                   (8,4),(8,5),(8,6),(8,7),(8,8),(8,9),(8,10),(8,11),(8,12),
                (9,3),(9,4),(9,5),(9,6),(9,7),(9,8),(9,9),(9,10),(9,11),(9,12),
       (10,2),(10,3),(10,4),(10,5),(10,6),(10,7),(10,8),(10,9),(10,10),(10,11),(10,12),
    (11,1),(11,2),(11,3),(11,4),(11,5),(11,6),(11,7),(11,8),(11,9),(11,10),(11,11),(11,12),
(12,0),(12,1),(12,2),(12,3),(12,4),(12,5),(12,6),(12,7),(12,8),(12,9),(12,10),(12,11),(12,12),
                                (13,4),(13,5),(13,6),(13,7),
                                   (14,4),(14,5),(14,6),
                                       (15,4),(15,5),
                                          (16,4),
]  # fmt: skip

# Starting positions of player 1
PLAYER1_STARTING_POSITIONS: list[BoardIndex] = [
(12,4),(12,5),(12,6),(12,7),(12,8),
    (13,4),(13,5),(13,6),(13,7),
       (14,4),(14,5),(14,6),
           (15,4),(15,5),
              (16,4),
]  # fmt: skip

# Starting positions of player 2
PLAYER2_STARTING_POSITIONS: list[BoardIndex] = [
            (0,12),
        (1,11),(1,12),
     (2,10),(2,11),(2,12),
  (3,9),(3,10),(3,11),(3,12),
(4,8),(4,9),(4,10),(4,11),(4,12),
]  # fmt: skip


# A movement is represented as a pair of board indices: (from, to)
type Movement = tuple[BoardIndex, BoardIndex]


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

    @classmethod
    def from_int32(cls, i: np.int32) -> "Position":
        return cls(i.item())

    def into_int32(self) -> np.int32:
        return np.int32(self.value)


@final
@dataclass
class Board:
    state: NDArray[np.int32]

    @classmethod
    def empty(cls) -> "Board":
        state = np.full((17, 17), Position.Invalid.into_int32(), dtype=np.int32)
        for idx in VALID_POSITIONS:
            state[idx] = Position.Empty.into_int32()
        return cls(state=state)

    @classmethod
    def two_players(cls) -> "Board":
        state = np.full((17, 17), Position.Invalid.into_int32(), dtype=np.int32)
        for idx in VALID_POSITIONS:
            state[idx] = Position.Empty.into_int32()
        for idx in PLAYER1_STARTING_POSITIONS:
            state[idx] = Position.Player1.into_int32()
        for idx in PLAYER2_STARTING_POSITIONS:
            state[idx] = Position.Player2.into_int32()
        return cls(state=state)

    def __getitem__(self, idx: BoardIndex) -> Position:
        return Position.from_int32(self.state[idx])

    def __setitem__(self, idx: BoardIndex, value: Position) -> None:
        self.state[idx] = value.into_int32()

    def print(self):
        for i, row in enumerate(self.state):
            print(
                f"{' ' * i}{''.join([str(Position.from_int32(cell)) for cell in row])}"
            )

    def apply_movement(self, movement: Movement) -> None:
        from_idx, to_idx = movement
        player = self[from_idx]
        # if (
        #     player == Position.Empty
        #     or player == Position.Invalid
        #     or self[to_idx] != Position.Empty
        # ):
        #     raise ValueError(f"Invalid movement from {from_idx} to {to_idx}")
        self[to_idx] = player
        self[from_idx] = Position.Empty


@final
class Agent:
    def __init__(self, player: Player):
        self.player = player
        self.board = Board.two_players()

    class Strategy:
        def select_move(self, movements: list[Movement]) -> int:
            raise NotImplementedError

    @final
    class BrownianStrategy(Strategy):
        @override
        def select_move(self, movements: list[Movement]) -> int:
            return random.randrange(0, len(movements))

    @final
    @dataclass(frozen=True)
    class AheadStrategy(Strategy):
        player: Player

        @override
        def select_move(self, movements: list[Movement]) -> int:
            movements_sorted = sorted(enumerate(movements), key=lambda mv: mv[1][1])
            match self.player:
                case Player.Player1:
                    if random.random() < 0.5:
                        return movements_sorted[-1][0]
                    else:
                        return movements_sorted[-2][0]

                case Player.Player2:
                    if random.random() < 0.5:
                        return movements_sorted[0][0]
                    else:
                        return movements_sorted[1][0]

    def decide_move(self, strategy: Strategy, movements: list[Movement]) -> int:
        return strategy.select_move(movements)
