import argparse
import asyncio
from dataclasses import dataclass
import json
from pprint import PrettyPrinter
import struct
import logging
from typing import final, override

from tictactoe import Agent, GameResult, GameResultDraw, GameResultVictory, Player


# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="[{asctime} {levelname}] {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Custom pretty printer for better readability of logs
printer = PrettyPrinter(indent=4, width=80, compact=True, sort_dicts=False)

# Set up command-line argument parser
parser = argparse.ArgumentParser(
    description="Tic-Tac-Toe Agent Client",
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
class ServerMessage:
    pass


@final
@dataclass(frozen=True)
class ServerMessageTest(ServerMessage):
    num: int


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
    available_moves: list[tuple[int, int]]


@final
@dataclass(frozen=True)
class ServerMessageMovement(ServerMessage):
    player: Player
    position: tuple[int, int]


@final
@dataclass(frozen=True)
class ServerMessageGameFinished(ServerMessage):
    result: GameResult


def parse_message(message_dict) -> ServerMessage:
    match message_dict.get("type"):
        case "test":
            num = message_dict["num"]
            return ServerMessageTest(num=num)

        case "assign":
            match message_dict["player"]:
                case "x":
                    player = Player.Cross
                case "o":
                    player = Player.Nought
                case _:
                    raise ValueError("Invalid player")
            return ServerMessageAssign(player=player)

        case "disconnect":
            return ServerMessageDisconnect()

        case "turn":
            available_moves = message_dict["available_moves"]
            return ServerMessageTurn(available_moves=available_moves)

        case "movement":
            player = Player.from_str(message_dict["player"])
            position: tuple[int, int] = message_dict["position"]
            return ServerMessageMovement(player=player, position=position)

        case "game_finished":
            result = message_dict["result"]
            match result["type"]:
                case "victory":
                    player = Player.from_str(result["player"])
                    result = GameResultVictory(player=player)
                case "draw":
                    result = GameResultDraw()
            return ServerMessageGameFinished(result=result)

        case _:
            raise ValueError(f"Unexpected message type: {message_dict.get('type')}")


# Client -> Server
class ClientMessage:
    def to_dict(self):
        return {}


@final
@dataclass(frozen=True)
class ClientMessageTest(ClientMessage):
    num: int

    @override
    def to_dict(self):
        return {"type": "test", **vars(self)}


@final
@dataclass(frozen=True)
class ClientMessageMovement(ClientMessage):
    position: tuple[int, int]

    @override
    def to_dict(self):
        return {"type": "movement", **vars(self)}


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

        message_dict = json.loads(message_json)
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

        # Game loop
        while True:
            message = await client.receive_message()

            match message:
                case ServerMessageTest(num):
                    num = message.num
                    message = ClientMessageTest(num=num)
                    logging.info(f"Sending message: {printer.pformat(message)}")
                    await client.send_message(message)

                case ServerMessageTurn(available_moves):
                    logging.debug("It's my turn")
                    position = agent.decide_move(available_moves)
                    logging.debug(f"Chosen move: {position}")
                    await client.send_message(ClientMessageMovement(position=position))

                case ServerMessageDisconnect():
                    logging.info("Disconnection signal received")
                    break

                case ServerMessageMovement(player, position):
                    logging.debug(f"Player {player} made move {position}")
                    agent.board[position] = player
                    print(f"{agent.board}\n")

                case ServerMessageGameFinished(result):
                    logging.info("Game finished")
                    match result:
                        case GameResultVictory(player):
                            logging.info(f"Player {player} is wins")
                        case GameResultDraw():
                            logging.info("It's a draw")
                    # Break out of game loop
                    break

                case _:
                    logging.error(f"Invalid message received: {message}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Client stopped by user.")
