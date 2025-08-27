from abc import ABC, abstractmethod
import logging
from typing import final, override

import numpy as np
from numpy.typing import NDArray

from client import (
    Client,
    ClientMessageChoice,
    GameResult,
    ServerMessage,
    ServerMessageDisconnect,
    ServerMessageGameFinished,
    ServerMessageMovement,
    ServerMessageTurn,
)

from sternhalma import (
    Board,
    Movement,
    Player,
)


class Agent(ABC):
    def __init__(self, player: Player):
        # Player assigned to agent
        self.player: Player = player
        # Board state
        self.board: Board = Board.two_players()

    async def play(self, client: Client) -> GameResult:
        logging.info("Agent started playing...")
        while True:
            match await client.receive_message():
                case ServerMessageTurn(movements):
                    logging.debug("It's my turn")
                    movement: list[list[int]] = self.decide_movement(movements).tolist()
                    logging.debug(f"Chosen movement: {movement}")
                    await client.send_message(ClientMessageChoice(movement))

                case ServerMessageMovement(player, indices):
                    logging.debug(f"Player {player} made move {indices}")
                    self.board.apply_movement(indices)
                    # agent.board.print()

                case ServerMessageGameFinished(result):
                    return result

                case ServerMessageDisconnect():
                    logging.error("Disconnection signal received mid game")
                    raise ConnectionAbortedError

                case ServerMessage():
                    pass

    def prepare_training(self):
        self.nn: None = None

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
class AgentDQN(Agent):
    @override
    def __init__(self, player: Player):
        # Parent constructor
        super().__init__(player)

        # Neural network
        self.nn = None

    @override
    def decide_movement(self, movements: NDArray[np.int_]) -> Movement:
        pass
