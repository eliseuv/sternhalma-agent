import asyncio

import cbor2
import struct
import logging
from typing import Any, final


from utils import printer


from protocol import (
    ClientMessage,
    ClientMessageHello,
    ClientMessageReconnect,
    ServerMessage,
    ServerMessageReject,
    ServerMessageWelcome,
)


@final
class Client:
    """Async TCP client for connecting to the game server"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        timeout: int = 30,  # 30s
        delay: float = 0.500,  # 500ms
        attempts: int = 20,
    ):
        """Connect to the game server.

        Args:
            host (str): Server host.
            port (int): Server port.
            timeout (int): Connection timeout.
            delay (float): Connection retry delay.
            attempts (int): Number of connection attempts.
        """

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

        self.session_id: str | None = None

    async def __aenter__(self):
        """Enter the async context manager"""

        logging.info(f"Connecting to server at {self.host}:{self.port}")
        for attempt in range(self.attempts):
            try:
                self.reader, self.writer = await asyncio.open_connection(
                    self.host, self.port
                )
                logging.info("Connection established successfully")

                # Send Hello immediately after connection
                if self.session_id:
                    logging.info(
                        f"Attempting to reconnect with session ID: {self.session_id}"
                    )
                    await self.send_message(ClientMessageReconnect(self.session_id))
                else:
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
        """Exit the async context manager"""

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
        """Receive a message from the server"""

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
        """Send a message to the server

        Args:
            message (ClientMessage): The message to send
        """

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

    async def handshake(self):
        """Perform the handshake with the server"""

        # Wait for "Welcome" message first
        message = await self.receive_message()
        match message:
            case ServerMessageWelcome(session_id):
                # The server accepts the connection and assigns a session ID
                logging.info(f"Welcome! Session ID: {session_id}")
                self.session_id = session_id
                return

            case ServerMessageReject(reason):
                # The server rejects the connection
                logging.error(f"Connection rejected: {reason}")
                self.session_id = None
                raise ConnectionRefusedError(f"Server rejected connection: {reason}")

            case _:
                # In case of an unexpected message, log it and raise an error
                logging.error(f"Expected Welcome message, got: {message}")
                raise ValueError(f"Unexpected message during handshake: {message}")
