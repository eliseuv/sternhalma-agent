import argparse
import asyncio
from dataclasses import dataclass
import json
import locale
from pprint import PrettyPrinter
import struct
import logging
from typing import Self, final, override


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
    help="Server address to connect to.",
)
_ = parser.add_argument(
    "-p",
    "--port",
    type=int,
    required=True,
    help="Server port to connect to.",
)

# Parse command-line arguments
args = parser.parse_args()
logging.debug(f"Arguments: {printer.pformat(vars(args))}")


# Server -> Client
class ClientMessage:
    pass


@final
@dataclass(frozen=True)
class TestClientMessage(ClientMessage):
    num: int


def parse_message(message_dict) -> ClientMessage:
    match message_dict.get("type"):
        case "test":
            num = message_dict["num"]
            return TestClientMessage(num=num)

        case _:
            raise ValueError(f"Unexpected message type: {message_dict.get['type']}")


# Client -> Server
class ServerMessage:
    def to_dict(self):
        return {}


@final
@dataclass(frozen=True)
class TestServerMessage(ServerMessage):
    num: int

    @override
    def to_dict(self):
        return {"type": "test", "num": self.num}


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

    async def receive_message(self) -> ClientMessage:
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

    async def send_message(self, message: ServerMessage):
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

            logging.debug(f"Message successfully sent")
            return  # Message sent successfully, exit loop

        else:
            raise ConnectionResetError(
                f"Unable to send message after {self.attempts} attempts"
            )


async def main():
    async with Client(args.host, args.port) as client:
        while True:
            message = await client.receive_message()
            logging.info(f"Received message: {printer.pformat(message)}")

            match message:
                case TestClientMessage(num):
                    num = message.num
                    message = TestServerMessage(num=num)
                    logging.info(f"Sending message: {printer.pformat(message)}")
                    await client.send_message(message)

                case _:
                    logging.error(f"Invalid message received: {message}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Client stopped by user.")
