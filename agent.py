from abc import ABC, abstractmethod
import logging
from typing import final, override

import numpy as np
from numpy.typing import NDArray

from client import (
    Client,
    ClientMessageChoice,
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

    async def play(self, client: Client):
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

                case ServerMessageGameFinished(winner):
                    logging.info(f"Game finished! Winner: {winner}")
                    # Break out of game loop
                    break

                case ServerMessageDisconnect():
                    logging.info("Disconnection signal received")
                    break

                case ServerMessage():
                    pass

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
