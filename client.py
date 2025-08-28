from abc import ABC
import asyncio
from dataclasses import dataclass
import cbor2
import struct
import logging
from typing import Any, final

import numpy as np
from numpy.typing import NDArray

from sternhalma import Player
from utils import printer


# Game result
class GameResult(ABC):
    """Abstract class for the result of a game"""

    pass


@final
@dataclass(frozen=True)
class GameResultMaxTurns(GameResult):
    """Game has reached its maximum number of turns

    Attributes:
        total_turns: Total number of turns played
    """

    total_turns: int


@final
@dataclass(frozen=True)
class GameResultFinished(GameResult):
    """The game has been played until completion

    Attributes:
        winner: Winner of the game
        total_turns: Total number of turns played
    """

    winner: Player
    total_turns: int


# Server -> Client
class ServerMessage(ABC):
    """Message from Server to Client"""

    pass


@final
@dataclass(frozen=True)
class ServerMessageAssign(ServerMessage):
    """Server assigns a player to client

    Attributes:
        player: Player the client was assign
    """

    player: Player


@final
@dataclass(frozen=True)
class ServerMessageDisconnect(ServerMessage):
    """Server requests that the client disconnects"""

    pass


@final
@dataclass(frozen=True)
class ServerMessageTurn(ServerMessage):
    """Server informs the client the it is their turn to play

    Attributes:
        movements: Available moves on the board
    """

    movements: NDArray[np.int_]


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


@final
@dataclass(frozen=True)
class ServerMessageGameFinished(ServerMessage):
    """Server informs the client that the game has finished and its result

    Attributes:
        winner: Player that won the game
        turns: Total number of turns played
    """

    result: GameResult


def parse_server_message(message: dict[str, Any]) -> ServerMessage:
    """Parse message received from the server from a dictionary to a `ServerMessage` type

    Args:
        message: Dictionary containing the message

    Returns:
        Parsed message

    Raises:
        ValueError: Invalid message type
    """
    match message.get("type"):
        # Player assignment message
        case "assign":
            return ServerMessageAssign(player=Player.from_str(message["player"]))

        # Disconnection request
        case "disconnect":
            return ServerMessageDisconnect()

        # It's the player's turn
        case "turn":
            return ServerMessageTurn(movements=np.array(message["movements"]))

        # Player made a movement
        case "movement":
            return ServerMessageMovement(
                player=Player.from_str(message["player"]),
                movement=np.array(message["movement"]),
            )

        # Game has finished
        case "game_finished":
            result = message["result"]
            match result["type"]:
                # Maximum number of turns reached
                case "max_turns":
                    result = GameResultMaxTurns(total_turns=result["total_turns"])
                # Game has a winner
                case "finished":
                    result = GameResultFinished(
                        winner=result["winner"], total_turns=result["total_turns"]
                    )
                case _:
                    raise ValueError(
                        f"Unexpected game result type: {result.get('type')}"
                    )
            return ServerMessageGameFinished(result=result)

        case _:
            raise ValueError(f"Unexpected message type: {message.get('type')}")


# Client -> Server
@dataclass(frozen=True)
class ClientMessage(ABC):
    """Message from Client to Server
    Every client message must provide a `type: str` field."""

    pass


@final
@dataclass(frozen=True)
class ClientMessageChoice(ClientMessage):
    """Client has chosen a movement from the list of available ones

    Attributes:
        movement: Chosen movement
    """

    movement: list[list[int]]
    type: str = "choice"


@final
class Client:
    def __init__(
        self,
        socket: str,
        delay: float = 0.500,  # 500ms
        attempts: int = 20,
        buf_size: int = 1024,
    ):
        # Socket connection parameters
        self.path = socket

        # Connection retry parameters
        self.delay = delay
        self.attempts = attempts

        # Stream reader and writer
        self.reader: asyncio.StreamReader = None  # pyright: ignore [reportAttributeAccessIssue]
        self.writer: asyncio.StreamWriter = None  # pyright: ignore [reportAttributeAccessIssue]

    async def __aenter__(self):
        logging.info(f"Connecting to server at {self.path}")
        for attempt in range(self.attempts):
            try:
                self.reader, self.writer = await asyncio.open_unix_connection(self.path)
                logging.info("Connection established successfully")
                return self

            # If the socket file is not found, log the error, wait and retry
            except FileNotFoundError:
                logging.error(
                    f"Socket file not found. Retrying {attempt + 1}/{self.attempts}"
                )
                await asyncio.sleep(self.delay)
                continue

            # If the connection is refused, log the error, wait and retry
            except ConnectionRefusedError:
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
            length_bytes = await self.reader.readexactly(4)
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
            message_bytes = await self.reader.readexactly(length)
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
        message = parse_server_message(message_dict)
        logging.debug(f"Received message: {printer.pformat(message)}")

        return message

    async def send_message(self, message: ClientMessage):
        logging.debug(f"Sending message to server: {printer.pformat(message)}")

        message_dict = vars(message)
        logging.debug(f"Message dict: {printer.pformat(message_dict)}")

        message_bytes = cbor2.dumps(message_dict)

        length: int = len(message_bytes)
        logging.debug(f"Message length: {length} bytes")

        length_bytes = struct.pack(">I", length)

        for _ in range(self.attempts):
            # Write the 4-byte length prefix and then the message payload
            try:
                self.writer.write(length_bytes)
                self.writer.write(message_bytes)
                # Ensure the data is actually sent
                await self.writer.drain()
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                logging.error(f"Error writing message to stream: {e}")
                continue  # Retry if it's a transient network error
            except Exception as e:
                logging.error(f"Unexpected error during message send: {e}")
                raise  # Re-raise unexpected errors

            logging.debug("Message successfully sent")
            return  # Message sent successfully, exit loop

        else:
            raise ConnectionResetError(
                f"Unable to send message after {self.attempts} attempts"
            )

    async def assign_player(self) -> Player:
        message = await self.receive_message()

        match message:
            case ServerMessageAssign(player):
                logging.info(f"Assigned player {player}")
                return player

            case _:
                logging.error(f"Invalid message received: {printer.pformat(message)}")
                raise ValueError(f"Invalid message received: {message}")
