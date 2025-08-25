import argparse
import asyncio
import logging

from agent import AgentBrownian
from client import Client
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
            await agent.play(client)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Client stopped by user.")
