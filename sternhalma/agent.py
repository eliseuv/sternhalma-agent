from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final, override

import numpy as np
from numpy.typing import NDArray

from .board import (
    PLAYER1_STARTING_POSITIONS,
    PLAYER2_STARTING_POSITIONS,
    Board,
    Movement,
    Player,
)


class Agent(ABC):
    def __init__(self, player: Player):
        self.player: Player = player
        self.board: Board = Board.two_players()

    @abstractmethod
    def decide_movement(self, movements: NDArray[np.int_]) -> Movement:
        pass


@final
class AgentConstant(Agent):
    @override
    def decide_movement(self, movements: NDArray[np.int_]) -> Movement:
        return movements[0]


@final
class AgentBrownian(Agent):
    @override
    def decide_movement(self, movements: NDArray[np.int_]) -> Movement:
        return movements[np.random.randint(0, len(movements))]


@final
class AgentTest(Agent):
    def __init__(self, player: Player):
        super().__init__(player)
        match player:
            case Player.Player1:
                self.goal = set(map(tuple, PLAYER2_STARTING_POSITIONS))
            case Player.Player2:
                self.goal = set(map(tuple, PLAYER1_STARTING_POSITIONS))

    @override
    def decide_movement(self, movements: NDArray[np.int_]) -> Movement:
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
