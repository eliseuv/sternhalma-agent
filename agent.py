from abc import ABC, abstractmethod
from collections.abc import Iterator
import random
from typing import final, override

import numpy as np
from numpy.typing import NDArray

from sternhalma import (
    PLAYER1_STARTING_POSITIONS,
    PLAYER2_STARTING_POSITIONS,
    Board,
    Movement,
    Player,
)


class Strategy(ABC):
    @abstractmethod
    def select_move(self, movements: NDArray[np.uintp]) -> int:
        pass


@final
class BrownianStrategy(Strategy):
    @override
    def select_move(self, movements: NDArray[np.uintp]) -> int:
        return random.randrange(0, len(movements))


@final
class AheadStrategy(Strategy):
    def __init__(self, player: Player):
        self.player = player
        match player:
            case Player.Player1:
                self.goal = PLAYER2_STARTING_POSITIONS
            case Player.Player2:
                self.goal = PLAYER1_STARTING_POSITIONS

    @override
    def select_move(self, movements: NDArray[np.uintp]) -> int:
        if random.random() < 0.5:
            return self.sort_movements_y(self.filter_out_goal(movements))[1][0]
        else:
            return random.randrange(0, len(movements))

    def filter_out_goal(self, movements: Iterator[Movement]):
        return filter(lambda mv: mv[0] not in self.goal, movements)

    def sort_movements_y(self, movements: Iterator[Movement]):
        return sorted(
            enumerate(movements),
            key=lambda mv: mv[1][1],
            reverse=(self.player == Player.Player2),
        )


@final
class Agent:
    def __init__(self, player: Player, strategy: Strategy):
        self.player = player
        self.strategy = strategy
        self.board = Board.two_players()

    def decide_move(self, movements: NDArray[np.uintp]) -> int:
        return self.strategy.select_move(movements)
