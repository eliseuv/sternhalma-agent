import logging
from abc import ABC, abstractmethod
from typing import final, override

import numpy as np
from numpy.typing import NDArray

from client.client import Client
from client.protocol import (
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
)


class Agent(ABC):
    def __init__(self):
        # Board state
        self.board: Board = Board.two_players()

    async def play(self, client: Client) -> GameResult:
        logging.info("Agent started playing...")
        while True:
            match await client.receive_message():
                case ServerMessageTurn(movements):
                    logging.debug("It's my turn")
                    movement_index: int = self.decide_movement(movements)
                    logging.debug(f"Chosen movement index: {movement_index}")
                    await client.send_message(ClientMessageChoice(movement_index))

                case ServerMessageMovement(player, indices):
                    logging.debug(f"Player {player} made move {indices}")
                    self.board.apply_movement(indices)
                    # agent.board.print()

                case ServerMessageGameFinished(result):
                    return result

                case ServerMessageDisconnect():
                    logging.error("Disconnection signal received mid game")
                    raise ConnectionAbortedError

                case ServerMessage() as unhandled:
                    logging.warning(f"Unhandled server message: {unhandled}")

    def prepare_training(self):
        self.nn: None = None

    @abstractmethod
    def decide_movement(self, movements: NDArray[np.int_]) -> int:
        pass


@final
class AgentConstant(Agent):
    @override
    def decide_movement(self, movements: NDArray[np.int_]) -> int:
        return 0


@final
class AgentBrownian(Agent):
    @override
    def decide_movement(self, movements: NDArray[np.int_]) -> int:
        return np.random.randint(0, len(movements))


@final
class AgentDQN(Agent):
    @override
    def __init__(self):
        # Parent constructor
        super().__init__()

        # Neural network
        self.nn = None

    @override
    def decide_movement(self, movements: NDArray[np.int_]) -> int:
        return 0
