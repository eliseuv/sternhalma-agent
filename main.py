import argparse
import asyncio
import logging


from agent import Agent, AgentBrownian
from client import Client
from protocol import GameResult, GameResultFinished, GameResultMaxTurns
from utils import printer


# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="[{asctime} {levelname}] {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set up command-line argument parser
parser = argparse.ArgumentParser(
    prog="SternhalmaAgent",
    description="Sternhalma player agent",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
# Game server socket
_ = parser.add_argument(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Game server host",
)
_ = parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Game server port",
)
# Training mode
_ = parser.add_argument(
    "--train",
    action="store_true",
    dest="training_mode",
    help="Enable agent training mode",
)


async def play(agent: Agent, client: Client):
    result = await agent.play(client)
    match result:
        case GameResultMaxTurns(total_turns, scores):
            logging.info(
                f"The game has reached its maximum number of turns {total_turns} with scores {scores}"
            )

        case GameResultFinished(winner, total_turns):
            logging.info(f"Game finished! Winner {winner} after {total_turns} turns")

        case GameResult():
            pass


async def main():
    # Parse command-line arguments
    args = parser.parse_args()
    logging.debug(f"Arguments: {printer.pformat(vars(args))}")

    host = str(args.host)
    port = int(args.port)
    training_mode = bool(args.training_mode)

    # Spawn client
    async with Client(host, port) as client:
        # Wait for player assignment from server
        await client.handshake()

        # Create agent
        agent = AgentBrownian()

        # Is the agent training or playing?
        if training_mode:
            pass

        else:
            await play(agent, client)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Client stopped by user.")
