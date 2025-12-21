import asyncio

import cbor2
import struct
import logging
from typing import Any, final


from sternhalma import Player
from utils import printer


from protocol import (
    ClientMessage,
    ClientMessageHello,
    ServerMessage,
    ServerMessageReject,
    ServerMessageWelcome,
)


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
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None

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

        if self.reader is None:
            raise ConnectionError("Client not connected")

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
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Message dict: {printer.pformat(message_dict)}")

        # Parse message
        message = ServerMessage.parse(message_dict)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Received message: {printer.pformat(message)}")

        return message

    async def send_message(self, message: ClientMessage):
        if self.writer is None:
            raise ConnectionError("Client not connected")

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Sending message to server: {printer.pformat(message)}")

        # Message is serialized as a dictionary
        message_dict = vars(message)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
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

    async def handshake(self) -> Player:
        # Wait for Welcome message first
        message = await self.receive_message()
        match message:
            case ServerMessageWelcome(session_id):
                logging.info(
                    f"Welcome! Session ID: {session_id}, Assigned player: {Player.Player1}"
                )
                return Player.Player1

            case ServerMessageReject(reason):
                logging.error(f"Connection rejected: {reason}")
                raise ConnectionRefusedError(f"Server rejected connection: {reason}")

            case _:
                # If we get an Assign message, it might be a reconnect scenario or protocol slight variation,
                # but per spec Hello -> Welcome. Let's handle Assign if it comes instead/after.
                # Actually, spec says Welcome has player.
                if logging.getLogger().isEnabledFor(logging.ERROR):
                    logging.error(
                        f"Expected Welcome message, got: {printer.pformat(message)}"
                    )
                raise ValueError(f"Unexpected message during handshake: {message}")
