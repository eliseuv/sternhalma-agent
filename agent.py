from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import final, override

from sternhalma import Board, Movement, Player


class Strategy(ABC):
    @abstractmethod
    def select_move(self, movements: list[Movement]) -> int:
        raise NotImplementedError("This is an abstract method")


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
        if random.random() < 0.2:
            movements_sorted = sorted(
                enumerate(movements),
                key=lambda mv: mv[1][1],
                reverse=(self.player == Player.Player2),
            )
            return movements_sorted[1][0]
        else:
            return random.randrange(0, len(movements))


@final
class Agent:
    def __init__(self, player: Player, strategy: Strategy = BrownianStrategy()):
        self.player = player
        self.board = Board.two_players()
        self.strategy = strategy

    def decide_move(self, movements: list[Movement]) -> int:
        return self.strategy.select_move(movements)
