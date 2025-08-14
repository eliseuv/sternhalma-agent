from abc import ABC, abstractmethod
from typing import final, override

import numpy as np
from numpy.typing import NDArray

from .board import (
    PLAYER1_STARTING_POSITIONS,
    PLAYER2_STARTING_POSITIONS,
    Board,
    Player,
)


class Strategy(ABC):
    @abstractmethod
    def select_move(self, movements: NDArray[np.int_]) -> NDArray[np.int_]:
        pass


@final
class ConstantStrategy(Strategy):
    @override
    def select_move(self, movements: NDArray[np.int_]) -> NDArray[np.int_]:
        return movements[0]


@final
class BrownianStrategy(Strategy):
    @override
    def select_move(self, movements: NDArray[np.int_]) -> NDArray[np.int_]:
        return movements[np.random.randint(0, len(movements))]


@final
class AheadStrategy(Strategy):
    def __init__(self, player: Player):
        self.player = player
        match player:
            case Player.Player1:
                self.goal = set(map(tuple, PLAYER2_STARTING_POSITIONS))
            case Player.Player2:
                self.goal = set(map(tuple, PLAYER1_STARTING_POSITIONS))

    @override
    def select_move(self, movements: NDArray[np.int_]) -> int:
        pass
        # return sorted(
        #     filter(
        #         lambda mv: ,
        #         enumerate(movements),
        #     ),
        #     key=lambda mv: min(map(hexagonal_metric, self.goal.T - mv[1][1])),
        # )[np.random.poisson(3)][0]
        # return sorted(
        #     filter(
        #         lambda mv: bool((self.goal.T != mv[1][0]).all(1).any()),
        #         enumerate(movements),
        #     ),
        #     key=lambda mv: np.linalg.norm(mv[1][0] - self.goal, axis=1).min(),
        #     reverse=True,
        # )[np.random.poisson(1)][0]


@final
class Agent:
    def __init__(self, player: Player, strategy: Strategy):
        self.player = player
        self.strategy = strategy
        self.board = Board.two_players()

    def decide_move(self, movements: NDArray[np.int_]) -> NDArray[np.int_]:
        return self.strategy.select_move(movements)
