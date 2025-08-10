from abc import ABC, abstractmethod
import argparse
import asyncio
from dataclasses import dataclass
import json
from pprint import PrettyPrinter
import struct
import logging
from typing import final, override

from sternhalma import Movement, Player
from agent import Agent, AheadStrategy


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
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host address",
)
_ = parser.add_argument(
    "-p",
    "--port",
    type=int,
    required=True,
    help="Port",
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
    movements: list[Movement]


@final
@dataclass(frozen=True)
class ServerMessageMovement(ServerMessage):
    player: Player
    movement: Movement


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
            movements = [
                list(map(tuple, movement)) for movement in message["movements"]
            ]
            return ServerMessageTurn(movements=movements)

        case "movement":
            movement = list(map(tuple, message["movement"]))
            return ServerMessageMovement(
                player=Player.from_str(message["player"]), movement=movement
            )

        case "game_finished":
            return ServerMessageGameFinished(winner=Player.from_str(message["winner"]))

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
        host: str,
        port: int,
        delay: float = 0.100,  # 100ms
        attempts: int = 10,
        buf_size: int = 1024,
    ):
        # Socket connection parameters
        self.host = host
        self.port = port

        # Connection retry parameters
        self.delay = delay
        self.attempts = attempts

        # Stream reader and writer
        self.reader: asyncio.StreamReader = None
        self.writer: asyncio.StreamWriter = None

    async def __aenter__(self):
        logging.info(f"Connecting to server at {self.host}:{self.port}")
        for attempt in range(self.attempts):
            try:
                self.reader, self.writer = await asyncio.open_connection(
                    self.host, self.port
                )
                logging.info("Connection established successfully")
                return self

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

    async def __aexit__(self, exc_type, exc_val, exc_tb):
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

        length: int = struct.unpack(">I", length_bytes)[0]
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
        message_json = message_bytes.decode("utf-8")
        logging.debug(f"Message JSON: {message_json}")

        message_dict: dict[str, object] = json.loads(message_json)
        logging.debug(f"Message dict: {printer.pformat(message_dict)}")

        # Parse message
        message = parse_message(message_dict)
        logging.debug(f"Received message: {printer.pformat(message)}")

        return message

    async def send_message(self, message: ClientMessage):
        logging.debug(f"Sending message to server: {printer.pformat(message)}")

        message_dict = message.to_dict()
        logging.debug(f"Message dict: {printer.pformat(message_dict)}")

        message_json = json.dumps(message_dict)
        logging.debug(f"Message JSON: {message_json}")

        message_bytes = message_json.encode("utf-8")

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
    async with Client(args.host, args.port) as client:
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
        agent = Agent(player)
        logging.info(f"Created agent for player {player}")
        logging.info(f"Agent strategy: {agent.strategy.__class__.__name__}")

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
