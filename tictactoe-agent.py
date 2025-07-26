import argparse
import json
import logging
import random
import socket
from enum import IntEnum
import time
from typing import Self, final, override

CONN_RETRY_DELAY_SECS = 1
MAX_CONN_ATTEMPTS = 10
BUFFER_SIZE = 1024


class Player(IntEnum):
    """Represents a player in the Tic-Tac-Toe game.

    Attributes:
        Empty: Unoccupied cell.
        Nought: Nought player (usually "o").
        Cross: Cross player (usually "x").
    """

    Empty = 0
    Nought = 1
    Cross = 2

    @override
    def __str__(self):
        match self:
            case Player.Empty:
                return "⬜"
            case Player.Nought:
                return "⭕"
            case Player.Cross:
                return "❌"

    @classmethod
    def from_str(cls, player_str: str | None):
        """Converts a string representation of a player to a Player enum.

        Args:
            player_str: String representation of the player, can be "o", "x", or None.

        Returns:
            Player: Corresponding Player enum value.

        Raises:
            ValueError: If the string does not match any known player representation.
        """

        match player_str:
            case "o":
                return Player.Nought
            case "x":
                return Player.Cross
            case None:
                return Player.Empty
            case _:
                raise ValueError(f"Unknown player string: {player_str}")

    def opponent(self):
        """Returns the opponent of the current player.

        Returns:
            Player: The opponent player.

        Raises:
            ValueError: If the current player is Empty, since it has no opponent.
        """

        match self:
            case Player.Nought:
                return Player.Cross
            case Player.Cross:
                return Player.Nought
            case Player.Empty:
                raise ValueError("Empty player has no opponent.")


@final
class Board:
    """Represents the Tic-Tac-Toe game board.

    Attributes:
        board: 2D list representing the game board, where each cell can be Empty, Nought, or Cross.
    """

    def __init__(self, board: list[list[Player]]):
        """Initializes the Tic-Tac-Toe board with a given state.

        Args:
            board: 2D list of Player enums representing the initial state of the board.
        """

        self.board = board

    @classmethod
    def empty(cls) -> Self:
        """Creates an empty Tic-Tac-Toe board.

        Returns:
            Board: An instance of the Board class with all cells empty.
        """

        return cls([[Player.Empty for _ in range(3)] for _ in range(3)])

    @classmethod
    def from_json(cls, board_json: list[list[str]]) -> Self:
        """Creates a Board instance from a JSON representation.

        Args:
            board_json: 2D list of strings representing the board state.

        Returns:
            Board: An instance of the Board class with the specified state.
        """

        return cls(list(map(lambda row: list(map(Player.from_str, row)), board_json)))

    @override
    def __str__(self):
        return "\n".join("".join(f"{cell}" for cell in row) for row in self.board)

    def __setitem__(self, position: tuple[int, int], player: Player):
        row, col = position
        if self.board[row][col] == Player.Empty.value:
            self.board[row][col] = player
        else:
            raise ValueError("Cell is already occupied.")


@final
class Agent:
    """Represents a Tic-Tac-Toe agent that plays the game.

    Attributes:
        player: The player type (Nought or Cross) that the agent will play as.
        board: The current state of the game board.
    """

    def __init__(self, player: Player):
        self.player = player
        self.board = Board.empty()

    def decide_opening(self) -> tuple[int, int]:
        chosen_opening = random.choice([(i, j) for i in range(3) for j in range(3)])
        logging.debug(f"Chosen opening move: {chosen_opening}")
        return chosen_opening

    def decide_move(self, available_actions: list[tuple[int, int]]) -> tuple[int, int]:
        chosen_move = random.choice(available_actions)
        logging.debug(f"Chosen move: {chosen_move}")
        return chosen_move

    def make_move(self, position: tuple[int, int]):
        self.board[position] = self.player


class Client:
    def __init__(self, s: socket.socket):
        self.s = s
        self.agent = Agent(Player.Empty)

    def connect_to_server(self, addr: str, port: int):
        """Connects client to the server.

        Args:
            addr: Server address
            port: Server port

        Returns:
            socket.socket: A connected socket to the server.

        Raises:
            ConnectionRefusedError: If the connection to the server fails after multiple attempts.
        """

        # Attempt to connect to the server
        logging.info(f"Connecting to server at {addr}:{port}...")
        for connection_attempt in range(MAX_CONN_ATTEMPTS):
            try:
                self.s.connect((addr, port))
                logging.info("Successfully connected to the server.")
                return
            except ConnectionRefusedError:
                logging.info(
                    f"Connection refused. Attempt {connection_attempt + 1}/{MAX_CONN_ATTEMPTS}. Retrying in {CONN_RETRY_DELAY_SECS}s..."
                )
                time.sleep(CONN_RETRY_DELAY_SECS)
                continue
        else:
            raise ConnectionRefusedError(
                f"Failed to connect to the server after {MAX_CONN_ATTEMPTS} attempts."
            )

    def receive_message(self):
        """Keeps receiving messages from the server until a valid JSON message is received.

        Returns:
            dict: The message received from the server.

        Raises:
            ConnectionResetError: If the connection to the server is lost.
        """

        while True:
            try:
                data = self.s.recv(BUFFER_SIZE)
            except ConnectionResetError:
                logging.critical("Connection lost to the server.")
                raise
            if not data:
                logging.critical("Connection closed by the server.")
                raise ConnectionResetError("Connection closed by the server.")

            try:
                message = json.loads(data.decode("utf-8"))
                logging.debug(f"Received message: {message}")
                return message
            except json.JSONDecodeError:
                # TODO: Warn server about malformed data
                logging.error(
                    f"Received malformed data from the server:\n{data.decode('utf-8')}\nRetrying..."
                )
                continue

    def send_message(self, message):
        """Sends a message to the server.

        Args:
            message: The message to send, as a dictionary.

        Raises:
            TypeError: If the message is not a serializable dictionary.
            ConnectionResetError: If the connection to the server is lost while sending the message.
        """

        try:
            self.s.sendall(json.dumps(message).encode("utf-8"))
            logging.debug(f"Sent message: {message}")
        except TypeError as e:
            logging.error(f"Failed to serialize message: {e}")
            raise TypeError("Message must be a serializable dictionary.")
        except (ConnectionResetError, BrokenPipeError):
            logging.critical("Connection lost while sending message.")
            raise ConnectionResetError("Connection lost while sending message.")

    def run(self):
        """Client that interacts with the server."""

        logging.debug("Waiting for initialization message from the server...")
        while True:
            try:
                message = self.receive_message()
            except ConnectionResetError:
                return

            match message.get("type"):
                case "connect":
                    match message["player"]:
                        case "o":
                            player = Player.Nought
                        case "x":
                            player = Player.Cross
                        case _:
                            logging.critical(
                                f"Unknown player type received from the server: {message['player']}"
                            )
                            return
                    logging.info(f"Assigned player: {player}")
                    self.agent.player = player
                    break
                case _:
                    logging.error(f"Received unexpected message: {message}")
                    continue

        # Make first move if player is Nought
        if self.agent.player == Player.Nought:
            logging.debug("Making the first move as Nought...")
            chosen_opening = self.agent.decide_opening()
            logging.info(f"Chosen opening move: {chosen_opening}")

            try:
                self.send_message(chosen_opening)
            except TypeError:
                return
            except ConnectionResetError:
                return

            self.agent.make_move(chosen_opening)
            logging.debug(f"\n{self.agent.board}")

        while True:
            try:
                message = self.receive_message()
            except ConnectionResetError:
                return

            match message.get("type"):
                case "your_turn":
                    opponent_move: tuple[int, int] = message["opponent_move"]
                    logging.info(f"Opponent's move: {opponent_move}")
                    self.agent.board[opponent_move] = self.agent.player.opponent()
                    logging.debug(f"\n{self.agent.board}")

                    logging.debug("It's my turn to play.")
                    available_moves: list[tuple[int, int]] = message["available_moves"]
                    logging.debug(f"Available moves: {available_moves}")

                    chosen_move = self.agent.decide_move(available_moves)
                    logging.info(f"Chosen move: {chosen_move}")

                    try:
                        self.send_message(chosen_move)
                    except TypeError:
                        return
                    except ConnectionResetError:
                        return

                    self.agent.make_move(chosen_move)
                    logging.debug(f"\n{self.agent.board}")

                case "game_finished":
                    logging.info("Game finished!")

                    result = message["result"]
                    match result["type"]:
                        case "draw":
                            logging.info("It's a draw!")

                        case "victory":
                            winner = Player.from_str(result["player"])
                            logging.info(f"Winner: {winner}!")

                        case _:
                            pass

                case "game_state":
                    logging.debug("Received game state update.")
                    self.agent.board = Board.from_json(message["board"])

                    logging.debug(f"Game state received:\n{self.agent.board}")

                case "disconnect":
                    logging.info("Received disconnect message from the server.")
                    return

                # TODO: Handle game errors more gracefully
                case "game_error":
                    error = message["error"]
                    match error:
                        case _:
                            logging.error(f"Game error received: {error}")
                    continue

                case _:
                    logging.error(f"Unexpected message received: {message}")
                    continue


def main():
    # Set up logging configuration
    logging.basicConfig(
        level=logging.DEBUG,
        format="[{asctime} {levelname}] {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse command-line arguments
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
        help="Server port to connect to.",
        required=True,
    )

    args = parser.parse_args()
    logging.debug(f"Arguments: {vars(args)}")
    addr: str = args.addr
    port: int = args.port

    try:
        # Connect to the Tic-Tac-Toe server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            client = Client(s)
            # Attempt to connect to the server
            logging.info(f"Connecting to server at {addr}:{port}...")
            client.connect_to_server(addr, port)

            # Launch the client to interact with the server
            client.run()

    except ConnectionRefusedError as e:
        logging.critical(f"Could not connect to the server: {e}")
        return
    except KeyboardInterrupt:
        logging.warning("\nConnection closed by user.")
        return
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")
        return


if __name__ == "__main__":
    main()
