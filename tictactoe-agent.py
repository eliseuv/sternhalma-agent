import argparse
from dataclasses import dataclass
import logging
from pprint import PrettyPrinter
import socket
import time
from typing import final

import cbor2

from tictactoe import Agent, Player

# Custom pretty printer for better readability of logs
printer = PrettyPrinter(indent=4, width=80, compact=True, sort_dicts=False)


@final
class Client:
    """Class to manage a client connection to a server.

    Attributes:
        addr (str): The server address to connect to.
        port (int): The server port to connect to.
        buf_size (int): Size of the buffer for receiving messages.
        delay (float): Delay in seconds between connection attempts.
        attempts (int): Number of connection attempts before giving up.
        _socket (socket.socket | None): The socket instance for the connection.
    """

    def __init__(
        self,
        addr: str,
        port: int,
        delay: float = 1,
        attempts: int = 10,
        buf_size: int = 1024,
    ):
        """Initializes the Client with connection parameters.

        Args:
            addr (str): The server address to connect to.
            port (int): The server port to connect to.
            buf_size (int): Size of the buffer for receiving messages. Default is 1024 bytes.
            delay (float): Delay in seconds between connection attempts. Default is 1 second.
            attempts (int): Number of connection attempts before giving up. Default is 10.
        """

        # Socket connection parameters
        self.addr = addr
        self.port = port

        # Connection retry parameters
        self.delay = delay
        self.attempts = attempts

        # Buffer size for receiving messages
        self.buf_size = buf_size

        # Socket instance
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def __enter__(self):
        """Establishes a socket connection to the server.

        Returns:
            Client: The instance of the Client class with an established socket connection.

        Raises:
            ConnectionRefusedError: If the connection to the server fails after multiple attempts.
        """

        logging.info(f"Connecting to server at {self.addr}:{self.port}")
        for attempt in range(self.attempts):
            try:
                self._socket.connect((self.addr, self.port))
                logging.info(
                    f"Successfully connected to server at {self.addr}:{self.port}"
                )
                return self

            # If the connection is refused, log the error, wait and retry
            except ConnectionRefusedError:
                logging.warning(
                    f"Connection refused. Retrying... ({attempt + 1}/{self.attempts})"
                )
                time.sleep(self.delay)
                continue

            # If any other exception occurs, log it and close the socket if it was created
            except Exception as e:
                logging.critical(f"An error occurred while connecting: {e}")
                # Close the socket if it was created
                if self._socket:
                    self._socket.close()
                raise

        # If all attempts fail, raise a ConnectionRefusedError
        else:
            raise ConnectionRefusedError(
                f"Failed to connect to the server after {self.attempts} attempts."
            )

    def __exit__(self, exc_type, exc_value, traceback):
        """Closes the socket connection when exiting the context manager.

        Args:
            exc_type: The type of the exception raised, if any.
            exc_value: The value of the exception raised, if any.
            traceback: The traceback object, if any.
        """
        # Close the socket connection if it exists
        if self._socket:
            self._socket.close()
            logging.debug("Socket connection closed.")

        # Handle exceptions that occurred during the context
        if exc_type:
            logging.error(f"An error occurred: {exc_value}")

        # Return False to propagate exceptions
        return False

    def receive_message(self):
        """Attempts to receive a message from the server, retrying if necessary.

        Returns:
            dict: The message received from the server, decoded from CBOR format.

        Raises:
            ConnectionRefusedError: If the connection to the server fails after multiple attempts.
        """
        logging.debug("Waiting for a message from the server...")
        for attempt in range(self.attempts):
            try:
                data = self._socket.recv(self.buf_size)
            except ConnectionResetError:
                logging.error(
                    f"Connection error. Retrying... ({attempt + 1}/{self.attempts})"
                )
                continue
            # Check if connection was closed by the server
            if not data:
                raise ConnectionResetError("Connection closed by the server.")
            # Attempt to decode the received data
            try:
                message = cbor2.loads(data)
            except cbor2.CBORDecodeError:
                logging.error(
                    f"Received malformed data from the server:\n{data}\nRetrying... ({attempt + 1}/{self.attempts})"
                )
                continue
            # Message successfully received and decoded
            logging.debug(f"Received message: {printer.pformat(message)}")
            return message
        else:
            raise ConnectionRefusedError(
                f"Failed to receive message from server after {self.attempts} attempts."
            )

    def send_message(self, message):
        """Sends a message to the server.

        Args:
            message (dict): The message to send, as a serializable dictionary.

        Raises:
            TypeError: If the message is not a serializable dictionary.
            ConnectionResetError: If the connection to the server is lost while sending the message.
        """

        logging.debug(f"Sending message:\n{message}")

        # Serialize the message
        try:
            data = cbor2.dumps(message)
        except TypeError as e:
            raise TypeError(f"Message could not be serialized: {e}")

        # Attempt to send the message to the server, retrying if necessary
        for attempt in range(self.attempts):
            try:
                self._socket.sendall(data)
            except (ConnectionResetError, BrokenPipeError) as e:
                logging.error(
                    f"Unable to send message: {e}\nRetrying... ({attempt + 1}/{self.attempts})"
                )
                continue
            # Message sent successfully, break the loop
            logging.debug("Message sent successfully.")
            return
        else:
            raise ConnectionRefusedError(
                f"Failed to send message to server after {self.attempts} attempts."
            )


@dataclass(frozen=True)
class InitializationMessage:
    player: Player


@dataclass(frozen=True)
class MovementMessage:
    player: Player
    position: tuple[int, int]


@dataclass(frozen=True)
class YourTurnMessage:
    available_moves: list[tuple[int, int]]


@dataclass(frozen=True)
class GameOverMessage:
    winner: Player | None


ServerMessage = (
    InitializationMessage | YourTurnMessage | MovementMessage | GameOverMessage
)


def parse_message(message) -> ServerMessage:
    message_type = message.get("type")
    match message_type:
        case "initialization":
            player = Player.from_str(message["player"])
            if player == Player.Empty:
                raise ValueError("Cannot initialize empty player")
            return InitializationMessage(player=player)

        case "your_turn":
            available_moves = message["available_moves"]
            return YourTurnMessage(available_moves=available_moves)

        case "movement":
            player = Player.from_str(message["player"])
            position = message["position"]
            return MovementMessage(player=player, position=position)

        case "game_over":
            result = message["result"]
            match result["type"]:
                case "draw":
                    return GameOverMessage(winner=None)

                case "victory":
                    player = Player.from_str(result["player"])
                    return GameOverMessage(winner=player)

                case _:
                    raise ValueError(
                        f"Unexpected game over result type: {result['type']}"
                    )

        case _:
            raise ValueError(f"Unexpected message type: {message_type}")


def main():
    # Set up logging configuration
    logging.basicConfig(
        level=logging.DEBUG,
        format="[{asctime} {levelname}] {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Tic-Tac-Toe Agent Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "-s",
        "--addr",
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
    logging.debug(f"Arguments: {vars(args)}")
    addr: str = args.addr
    port: int = args.port

    try:
        with Client(addr, port) as client:
            logging.debug("Waiting for inialization message from the server...")
            while True:
                message = client.receive_message()
                try:
                    message = parse_message(message)
                except ValueError as e:
                    logging.error(f"Failed to parse message: {e}")
                    continue
                match message:
                    case InitializationMessage(player):
                        logging.info(f"Assigned player: {player}")
                        break
                    case _:
                        logging.error(f"Received unexpected message: {message}")
                        continue

            logging.debug("Creating agent...")
            agent = Agent(player)

            # Enter game loop
            logging.debug("Entering game loop...")
            while True:
                # Wait for a message from the server
                message = client.receive_message()
                try:
                    message = parse_message(message)
                except ValueError as e:
                    logging.error(f"Failed to parse message: {e}")
                    continue
                logging.debug(f"Parsed message: {printer.pformat(message)}")

                match message:
                    case YourTurnMessage(available_moves):
                        logging.info("It's your turn to make a move.")
                        logging.debug(f"Available moves: {available_moves}")
                        chosen_move = agent.decide_move(available_moves)
                        logging.info(f"Chosen move: {chosen_move}")
                        client.send_message(
                            {"type": "movement", "position": chosen_move}
                        )

                    case MovementMessage(player, pos):
                        logging.info(f"Player {player} made move {pos}")
                        agent.board[pos] = player
                        logging.debug(f"Current board state:\n{agent.board}")

                    case GameOverMessage(winner):
                        logging.info("Game over.")
                        if winner:
                            logging.info(f"Winner: {winner}")
                        else:
                            logging.info("The game ended in a draw.")
                        return

                    case _:
                        logging.error(f"Received unexpected message: {message}")
                        continue

    except KeyboardInterrupt:
        logging.info("Client interrupted by user.")
        return

    except Exception as e:
        logging.critical(f"An error occurred: {e}")
        return


if __name__ == "__main__":
    main()
