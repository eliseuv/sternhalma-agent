from abc import ABC, abstractmethod
import argparse
import asyncio
from dataclasses import dataclass
import cbor2
from pprint import PrettyPrinter
import struct
import logging
from typing import final, override

import numpy as np
from numpy.typing import NDArray

from sternhalma import Player
from agent import Agent, BrownianStrategy


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[{asctime} {levelname}] {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Custom pretty printer for better readability of logs
printer = PrettyPrinter(indent=4, width=80, compact=True, sort_dicts=False)

# Set up command-line argument parser
parser = argparse.ArgumentParser(
    description="Sternhalma Agent Client",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
_ = parser.add_argument(
    "--socket",
    type=str,
    help="Unix socket path",
)

# Parse command-line arguments
args = parser.parse_args()
logging.debug(f"Arguments: {printer.pformat(vars(args))}")


# Server -> Client
class ServerMessage(ABC):
    pass


@final
@dataclass(frozen=True)
class ServerMessageAssign(ServerMessage):
    player: Player


@final
@dataclass(frozen=True)
class ServerMessageDisconnect(ServerMessage):
    pass


@final
@dataclass(frozen=True)
class ServerMessageTurn(ServerMessage):
    movements: NDArray[np.int_]


@final
@dataclass(frozen=True)
class ServerMessageMovement(ServerMessage):
    player: Player
    movement: NDArray[np.int_]


@final
@dataclass(frozen=True)
class ServerMessageGameFinished(ServerMessage):
    winner: Player


def parse_message(message: dict[str, object]) -> ServerMessage:
    match message.get("type"):
        case "assign":
            match message["player"]:
                case "1":
                    return ServerMessageAssign(player=Player.Player1)
                case "2":
                    return ServerMessageAssign(player=Player.Player2)
                case _:
                    raise ValueError("Invalid player")

        case "disconnect":
            return ServerMessageDisconnect()

        case "turn":
            return ServerMessageTurn(movements=np.array(message["movements"]))

        case "movement":
            return ServerMessageMovement(
                player=Player.from_str(str(message["player"])),
                movement=np.array(message["movement"]),
            )

        case "game_finished":
            return ServerMessageGameFinished(
                winner=Player.from_str(str(message["winner"]))
            )

        case _:
            raise ValueError(f"Unexpected message type: {message.get('type')}")


# Client -> Server
class ClientMessage(ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, object]:
        pass


@final
@dataclass(frozen=True)
class ClientMessageTest(ClientMessage):
    num: int

    @override
    def to_dict(self) -> dict[str, object]:
        return {"type": "test", **vars(self)}


@final
@dataclass(frozen=True)
class ClientMessageChoice(ClientMessage):
    index: int

    @override
    def to_dict(self) -> dict[str, object]:
        return {"type": "choice", **vars(self)}


@final
class Client:
    def __init__(
        self,
        socket: str,
        delay: float = 0.100,  # 100ms
        attempts: int = 10,
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

        length = int(struct.unpack(">I", length_bytes)[0])
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
        message_dict: dict[str, object] = cbor2.loads(message_bytes)
        logging.debug(f"Message dict: {printer.pformat(message_dict)}")

        # Parse message
        message = parse_message(message_dict)
        logging.debug(f"Received message: {printer.pformat(message)}")

        return message

    async def send_message(self, message: ClientMessage):
        logging.debug(f"Sending message to server: {printer.pformat(message)}")

        message_dict = message.to_dict()
        logging.debug(f"Message dict: {printer.pformat(message_dict)}")

        message_bytes = cbor2.dumps(message_dict)

        length: int = len(message_bytes)
        logging.debug(f"Message length: {length} bytes")

        length_bytes = struct.pack(">I", length)

        for attempt in range(self.attempts):
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


async def main():
    async with Client(args.socket) as client:
        # Wait for assignment message
        while True:
            message = await client.receive_message()
            match message:
                case ServerMessageAssign(player):
                    logging.info(f"Assigned player {player}")
                    player = player
                    break

                case _:
                    logging.error(
                        f"Invalid message received: {printer.pformat(message)}"
                    )
                    continue

        # Create game agent
        agent = Agent(player, BrownianStrategy())
        logging.info(f"Created agent with strategy: {agent.strategy}")

        # Game loop
        while True:
            message = await client.receive_message()

            match message:
                case ServerMessageTurn(movements):
                    logging.debug("It's my turn")
                    index = agent.decide_move(movements)
                    logging.debug(f"Chosen move: {movements[index]}")
                    await client.send_message(ClientMessageChoice(index=index))

                case ServerMessageMovement(player, indices):
                    logging.debug(f"Player {player} made move {indices}")
                    agent.board.apply_movement(indices)
                    # agent.board.print()

                case ServerMessageGameFinished(winner):
                    logging.info(f"Game finished! Winner: {winner}")
                    # Break out of game loop
                    break

                case ServerMessageDisconnect():
                    logging.info("Disconnection signal received")
                    break

                case _:
                    logging.error(f"Invalid message received: {message}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Client stopped by user.")
