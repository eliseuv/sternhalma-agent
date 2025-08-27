import argparse
import asyncio
import logging
from unittest import result

from agent import Agent, AgentBrownian
from client import Client, GameResult, GameResultFinished, GameResultMaxTurns
from utils import printer


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
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
    "--socket",
    type=str,
    required=True,
    help="Game server socket path",
)
# Agent mode
_ = parser.add_argument(
    "--train",
    action="store_true",
    dest="training_mode",
    help="Enable agent training mode",
)


async def play(agent: Agent, client: Client):
    result = await agent.play(client)
    match result:
        case GameResultMaxTurns(total_turns):
            logging.info(
                f"The game has reached its maximum number of turns: {total_turns}"
            )

        case GameResultFinished(winner, total_turns):
            logging.info(f"Game finshied! Winner {winner} after {total_turns} turns")

        case GameResult():
            pass


async def main():
    # Parse command-line arguments
    args = parser.parse_args()
    logging.debug(f"Arguments: {printer.pformat(vars(args))}")
    socket = str(args.socket)
    training_mode = bool(args.training_mode)

    # Connect to server
    async with Client(socket) as client:
        # Wait for player assignment from server
        player = await client.assign_player()

        # Create agent
        agent = AgentBrownian(player)

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
