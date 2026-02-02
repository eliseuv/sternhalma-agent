from abc import ABC
from dataclasses import dataclass
from typing import Any, final, override

import numpy as np
from numpy.typing import NDArray

from sternhalma import Player, Scores


# Game result
class GameResult(ABC):
    """Abstract class for the result of a game"""

    @classmethod
    def parse(cls, result: dict[str, Any]) -> "GameResult":
        match result["type"]:
            # Maximum number of turns reached
            case "max_turns":
                return GameResultMaxTurns.parse(result)

            # Game has a winner
            case "finished":
                return GameResultFinished.parse(result)

            case _:
                raise ValueError(f"Unexpected game result type: {result.get('type')}")


@final
@dataclass(frozen=True)
class GameResultMaxTurns(GameResult):
    """Game has reached its maximum number of turns

    Attributes:
        total_turns: Total number of turns played
        scores: Dictionary containing the scores of all players
    """

    total_turns: int
    scores: Scores

    @override
    @classmethod
    def parse(cls, result: dict[str, Any]) -> "GameResult":
        return cls(
            total_turns=result["total_turns"],
            scores=result["scores"],
        )


@final
@dataclass(frozen=True)
class GameResultFinished(GameResult):
    """The game has been played until completion

    Attributes:
        winner: Winner of the game
        total_turns: Total number of turns played
        scores: Dictionary containing the scores of all players
    """

    winner: Player
    total_turns: int
    scores: Scores

    @override
    @classmethod
    def parse(cls, result: dict[str, Any]) -> "GameResult":
        return cls(
            winner=Player(result["winner"]),
            total_turns=result["total_turns"],
            scores=result["scores"],
        )


# Server -> Client
class ServerMessage(ABC):
    """Message from Server to Client"""

    @classmethod
    def parse(cls, message: dict[str, Any]) -> "ServerMessage":
        match message.get("type"):
            # Server welcomes the client
            case "welcome":
                return ServerMessageWelcome.parse(message)

            # Server rejects the client
            case "reject":
                return ServerMessageReject.parse(message)

            # Disconnection request
            case "disconnect":
                return ServerMessageDisconnect.parse(message)

            # It's the player's turn
            case "turn":
                return ServerMessageTurn.parse(message)

            # Player made a movement
            case "movement":
                return ServerMessageMovement.parse(message)

            # Game has finished
            case "game_finished":
                return ServerMessageGameFinished.parse(message)

            case _:
                raise ValueError(f"Unexpected message type: {message.get('type')}")


@final
@dataclass(frozen=True)
class ServerMessageWelcome(ServerMessage):
    """Server accepts the connection from the client and assigns a session ID

    Attributes:
        session_id: Unique session identifier
    """

    session_id: str

    @override
    @classmethod
    def parse(cls, message: dict[str, Any]) -> "ServerMessage":
        return cls(
            session_id=message["session_id"],
        )


@final
@dataclass(frozen=True)
class ServerMessageReject(ServerMessage):
    """Server rejects the connection from the client

    Attributes:
        reason: Reason for rejection
    """

    reason: str

    @override
    @classmethod
    def parse(cls, message: dict[str, Any]) -> "ServerMessage":
        return cls(reason=message["reason"])


@final
@dataclass(frozen=True)
class ServerMessageDisconnect(ServerMessage):
    """Server requests that the client disconnects"""

    @override
    @classmethod
    def parse(cls, message: dict[str, Any]) -> "ServerMessage":
        return cls()


@final
@dataclass(frozen=True)
class ServerMessageTurn(ServerMessage):
    """Server informs the client the it is their turn to play

    Attributes:
        movements: Available moves on the board
    """

    movements: NDArray[np.int_]

    @override
    @classmethod
    def parse(cls, message: dict[str, Any]) -> "ServerMessage":
        return cls(movements=np.array(message["movements"]))


@final
@dataclass(frozen=True)
class ServerMessageMovement(ServerMessage):
    """Server informs that client that a movement was made on the board

    Attributes:
        player: Player that made the movement
        movement: Movement made
    """

    player: Player
    movement: NDArray[np.int_]
    scores: Scores

    @override
    @classmethod
    def parse(cls, message: dict[str, Any]) -> "ServerMessage":
        return cls(
            player=Player(message["player"]),
            movement=np.array(message["movement"]),
            scores=message["scores"],
        )


@final
@dataclass(frozen=True)
class ServerMessageGameFinished(ServerMessage):
    """Server informs the client that the game has finished and its result

    Attributes:
        winner: Player that won the game
        turns: Total number of turns played
    """

    result: GameResult

    @override
    @classmethod
    def parse(cls, message: dict[str, Any]) -> "ServerMessage":
        return cls(result=GameResult.parse(message["result"]))


# Client -> Server
@dataclass(frozen=True)
class ClientMessage(ABC):
    """Message from Client to Server
    Every client message must provide a `type: str` field."""

    pass


@final
@dataclass(frozen=True)
class ClientMessageHello(ClientMessage):
    """Client initiates a new session"""

    type: str = "hello"


@final
@dataclass(frozen=True)
class ClientMessageReconnect(ClientMessage):
    """Client requests to reconnect to an existing session

    Attributes:
        session_id: Session ID to reconnect to
    """

    session_id: str
    type: str = "reconnect"


@final
@dataclass(frozen=True)
class ClientMessageChoice(ClientMessage):
    """Client has chosen a movement index from the list of available ones

    Attributes:
        movement_index: Index of the chosen movement
    """

    movement_index: int
    type: str = "choice"
