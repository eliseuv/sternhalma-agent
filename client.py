from abc import ABC
import asyncio
from dataclasses import dataclass
import cbor2
import struct
import logging
from typing import Any, final, override

import numpy as np
from numpy.typing import NDArray

from sternhalma import Player, Scores
from utils import printer


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
            winner=Player.from_str(result["winner"]),
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

            # Player assignment message
            case "assign":
                return ServerMessageAssign.parse(message)

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
class ServerMessageAssign(ServerMessage):
    """Server assigns a player to client

    Attributes:
        player: Player the client was assign
    """

    player: Player

    @override
    @classmethod
    def parse(cls, message: dict[str, Any]) -> "ServerMessage":
        return cls(player=Player.from_str(message["player"]))


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
            player=Player.from_str(message["player"]),
            movement=np.array(message["movement"]),
            scores=message["scores"],
        )


@final
@dataclass(frozen=True)
class ServerMessageWelcome(ServerMessage):
    """Server welcomes the client and assigns a session ID and player

    Attributes:
        session_id: Unique session identifier
        player: Player assigned to the client
    """

    session_id: str
    player: Player

    @override
    @classmethod
    def parse(cls, message: dict[str, Any]) -> "ServerMessage":
        return cls(
            session_id=message["session_id"],
            player=Player.from_str(message["player"]),
        )


@final
@dataclass(frozen=True)
class ServerMessageReject(ServerMessage):
    """Server rejects the connection

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


@final
class Client:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        timeout: int = 30,
        delay: float = 0.500,  # 500ms
        attempts: int = 20,
        buf_size: int = 1024,
    ):
        # Socket connection parameters
        self.host = host
        self.port = port

        self.timeout = timeout

        # Connection retry parameters
        self.delay = delay
        self.attempts = attempts

        # Stream reader and writer
        self.reader: asyncio.StreamReader = None  # pyright: ignore [reportAttributeAccessIssue]
        self.writer: asyncio.StreamWriter = None  # pyright: ignore [reportAttributeAccessIssue]

    async def __aenter__(self):
        logging.info(f"Connecting to server at {self.host}:{self.port}")
        for attempt in range(self.attempts):
            try:
                self.reader, self.writer = await asyncio.open_connection(
                    self.host, self.port
                )
                logging.info("Connection established successfully")
                
                # Send Hello immediately after connection
                await self.send_message(ClientMessageHello())
                
                return self

            # If the connection fails, log the error, wait and retry
            except (ConnectionRefusedError, OSError):
                logging.error(
                    f"Connection refused. Retrying {attempt + 1}/{self.attempts}"
                )
                await asyncio.sleep(self.delay)
                continue

            except Exception as e:
                logging.critical(f"Failed to connect: {e}")
                raise

        # If all attempts fail, raise a ConnectionRefusedError
        else:
            raise ConnectionRefusedError(
                f"Failed to connect to the server after {self.attempts} attempts."
            )

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # pyright: ignore [reportMissingParameterType, reportUnknownParameterType]
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
            logging.info("Connection closed")

        # Handle exceptions that occurred during the context
        if exc_type:
            logging.error(f"An error occurred: {exc_val}")

        # Return False to propagate exceptions
        return False

    async def receive_message(self) -> ServerMessage:
        logging.debug("Waiting for server...")

        # Read the 4-byte length prefix
        try:
            length_bytes = await asyncio.wait_for(
                self.reader.readexactly(4), timeout=self.timeout
            )
        except asyncio.IncompleteReadError as e:
            bytes_read = len(e.partial)
            if bytes_read == 0:
                raise ConnectionResetError("Connnection closed by the server")
            else:
                raise

        length = int.from_bytes(length_bytes)
        logging.debug(f"Message length: {length} bytes")

        # Read the actual message payload
        try:
            message_bytes = await asyncio.wait_for(
                self.reader.readexactly(length), timeout=self.timeout
            )
        except asyncio.IncompleteReadError as e:
            bytes_read = len(e.partial)
            if bytes_read == 0:
                raise ConnectionResetError("Connnection closed by the server")
            else:
                raise

        # Decode message
        message_dict: dict[str, Any] = cbor2.loads(message_bytes)
        logging.debug(f"Message dict: {printer.pformat(message_dict)}")

        # Parse message
        message = ServerMessage.parse(message_dict)
        logging.debug(f"Received message: {printer.pformat(message)}")

        return message

    async def send_message(self, message: ClientMessage):
        logging.debug(f"Sending message to server: {printer.pformat(message)}")

        # Message is serialized as a dictionary
        message_dict = vars(message)
        logging.debug(f"Message dict: {printer.pformat(message_dict)}")

        # Binary message
        message_bytes = cbor2.dumps(message_dict)

        # Calculate message length
        length: int = len(message_bytes)
        logging.debug(f"Message length: {length} bytes")
        length_bytes = struct.pack(">I", length)

        # Write the 4-byte length prefix and then the message payload
        self.writer.write(length_bytes)
        self.writer.write(message_bytes)

        # Ensure the data is actually sent
        await asyncio.wait_for(self.writer.drain(), timeout=self.timeout)

        logging.debug("Message successfully sent")

    async def assign_player(self) -> Player:
        # Wait for Welcome message first
        message = await self.receive_message()
        match message:
            case ServerMessageWelcome(session_id, player):
                logging.info(f"Welcome! Session ID: {session_id}, Assigned player: {player}")
                return player
                
            case ServerMessageReject(reason):
                logging.error(f"Connection rejected: {reason}")
                raise ConnectionRefusedError(f"Server rejected connection: {reason}")
                
            case _:
                # If we get an Assign message, it might be a reconnect scenario or protocol slight variation,
                # but per spec Hello -> Welcome. Let's handle Assign if it comes instead/after.
                # Actually, spec says Welcome has player.
                logging.error(f"Expected Welcome message, got: {printer.pformat(message)}")
                raise ValueError(f"Unexpected message during handshake: {message}")
