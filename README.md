# Sternhalma Agent

This project aims to implement the AlphaZero algorithm from scratch to master the game of **Sternhalma** (commonly known as Chinese Checkers).

The agent is designed to interact with a Sternhalma server using a custom CBOR-based protocol.

## Project Goal

The primary objective is to build a reinforcement learning agent capable of achieving high-level play in Sternhalma without human knowledge, using:

- **Monte Carlo Tree Search (MCTS)** for lookahead planning.
- **Deep Neural Networks (ResNet)** for evaluating board states and predicting move policies.
- **Self-Play** for iterative improvement.

## Setup & Usage

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

### Installation

```bash
# Install dependencies
uv sync
```

### Running Tests

To verify the installation and current functionality:

```bash
uv run pytest
```

### Running the Agent

This agent requires a running instance of the Sternhalma server. You can find the server implementation and setup instructions here: [sternhalma-server](https://github.com/eliseuv/sternhalma-server).

To start the agent and connect to a server:

```bash
uv run main.py --host 127.0.0.1 --port 8080
```

**Arguments:**

- `--host`: The hostname or IP address of the game server (default: `127.0.0.1`).
- `--port`: The port number the server is listening on (default: `8080`).

## Project Structure

- `sternhalma.py`: Core game logic, board state, and coordinate systems.
- `alphazero.py`: Neural network architecture and tensor conversions.
- `client.py`: Async TCP client for connecting to the game server.
- `protocol.py`: Protocol message definitions (Server/Client messages).
- `agent.py`: Abstract agent definition and basic implementations (Random, Constant).

## Current Progress

- **Core Game Logic**: Complete implementation of the Sternhalma board, rules, and metrics in `sternhalma.py`. The module is fully documented.
- **Networking**: connection handling and protocol implementation in `client.py` and `protocol.py`. Supports asynchronous communication with the game server.
- **Neural Network**: Basic AlphaZero-style architecture (ResNet backbone, Policy Head, Value Head) implemented using PyTorch in `alphazero.py`.
- **State Representation**: Canonical board representation and tensor conversion logic (`from_state`) dealing with player perspectives and rotational symmetry.
- **Testing**: Integration test suite set up with `pytest` (`tests/test_integration.py`) covering the client-server handshake and game flow.
- **Dependency Management**: Project dependencies managed via `uv` and `pyproject.toml`.
