from dataclasses import dataclass
from enum import IntEnum
import random
from typing import final


class Player(IntEnum):
    Player1 = 1
    Player2 = 2

    @classmethod
    def from_str(cls, player_str: str):
        match player_str:
            case "1":
                return Player.Player1
            case "2":
                return Player.Player2
            case _:
                raise ValueError(f"Unknown player string: {player_str}")


@dataclass(frozen=True)
class GameResult:
    winner: Player


# Axial index in the hexagonal board
type BoardIndex = tuple[int, int]


# A movement is represented as a pair of board indices: (from, to)
type Movement = tuple[BoardIndex, BoardIndex]


@final
class Agent:
    def __init__(self, player: Player):
        self.player = player

    def decide_move(self, movements: list[Movement]) -> int:
        n = len(movements)
        chosen_move = random.randrange(0, n)
        return chosen_move
