from typing import override
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from sternhalma import Board, Position


def from_state(board: Board, current_player: int, device: str = "cuda") -> T.Tensor:
    """
    Converts the board state to a canonical tensor representation for neural network input.

    The representation is always relative to the `current_player`. This allows the network
    to learn a single policy for "friendly" vs "enemy" pieces regardless of which player it is.

    - If `current_player` is Player 1: No rotation.
    - If `current_player` is Player 2: The board is rotated 180 degrees so that Player 2
      appears to be playing from the same perspective as Player 1 (e.g. bottom to top).

    Args:
        board: The current board state.
        current_player: The player whose perspective the board should be viewed from (1 or 2).
        device: The device (e.g., "cpu", "cuda") where the tensor will be allocated.

    Returns:
        A tensor of shape (1, 3, 17, 17) ready for the network:
        - Channel 0: Binary mask for "friendly" pieces (current_player).
        - Channel 1: Binary mask for "enemy" pieces (opponent).
        - Channel 2: Binary mask for all valid board positions (board geometry).
    """
    tensor = np.zeros((3, 17, 17), dtype=np.float32)

    # Determine masks based on perspective
    # Channel 0 is always "me", Channel 1 is always "opponent"
    if current_player == 1:
        friend_mask = board.state == Position.Player1
        enemy_mask = board.state == Position.Player2
        should_rotate = False  # Player 1 plays in default orientation
    else:
        friend_mask = board.state == Position.Player2
        enemy_mask = board.state == Position.Player1
        should_rotate = True  # Player 2 board is rotated to match P1 perspective

    # Set masks for each channel
    tensor[0] = friend_mask.astype(np.float32)
    tensor[1] = enemy_mask.astype(np.float32)
    # Channel 2 is invariant (just valid positions)
    tensor[2] = (board.state != Position.Invalid).astype(np.float32)

    # Apply 180-degree rotation for Player 2 to canonicalize the board state
    # This effectively swaps the start/target triangles relative to the grid center
    if should_rotate:
        tensor = np.rot90(tensor, k=2, axes=(1, 2))

    # Convert to torch tensor, add batch dimension (N=1), and move to device
    return T.from_numpy(tensor.copy()).unsqueeze(0).to(device)


class ResBlock(nn.Module):
    """
    Standard Residual Block implementation.
    Consists of two 3x3 convolutional layers with batch normalization and a skip connection.
    """

    def __init__(self, num_channels: int) -> None:
        """
        Initializes the ResBlock with the specified number of channels.

        Args:
            num_channels (int): Number of input and output channels for the convolutions.
        """
        super().__init__()

        # First convolutional layer: 3x3 kernel, padding=1 to keep spatial dimensions constant.
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        # Batch normalization to stabilize and accelerate training.
        self.bn1 = nn.BatchNorm2d(num_channels)

        # Second convolutional layer: identical to the first.
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        # Batch normalization for the second convolution's output.
        self.bn2 = nn.BatchNorm2d(num_channels)

    @override
    def forward(self, x: T.Tensor) -> T.Tensor:
        """
        Forward pass through the residual block.

        Args:
            x (T.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            T.Tensor: Output tensor after applying the residual mapping.
        """
        # Save the input for the skip connection.
        residual = x

        # First block: Convolution -> Batch Norm -> ReLU.
        x = F.relu(self.bn1(self.conv1(x)))

        # Second block: Convolution -> Batch Norm.
        x = self.bn2(self.conv2(x))

        # Add the original input (residual) to the processed features.
        x += residual

        # Final ReLU activation.
        return F.relu(x)


class SternhalmaZero(nn.Module):
    """
    AlphaZero-inspired neural network architecture for Sternhalma.
    Consists of a shared residual backbone followed by separate policy and value heads.
    """

    def __init__(self, board_size: int, num_actions: int, num_res_blocks: int = 10):
        """
        Initializes the SternhalmaZero model.

        Args:
            board_size (int): The dimension of the square board representation.
            num_actions (int): Total number of possible actions in the game.
            num_res_blocks (int): Number of residual blocks in the backbone.
        """
        super().__init__()

        # Initial Convolutional Block: Extracts low-level features from the input state.
        self.start_conv = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.start_bn = nn.BatchNorm2d(128)

        # Backbone: A sequence of residual blocks to extract deep features.
        self.backbone = nn.Sequential(*[ResBlock(128) for _ in range(num_res_blocks)])

        # Policy Head: Predicts the probability distribution over all possible actions.
        # Outputs a vector of size `num_actions` with the log-probabilities (priors) of each action.
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, num_actions)

        # Value Head: Estimates the value of the current board state.
        # Outputs a scalar value typically interpreted in the range [-1, 1].
        self.value_conv = nn.Conv2d(128, 3, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    @override
    def forward(self, x: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        """
        Forward pass through the network to compute policy logits and state value.

        Args:
            x (T.Tensor): Input tensor of shape (N, 3, board_size, board_size).

        Returns:
            tuple[T.Tensor, T.Tensor]: A tuple containing:
                - policy: Logits for each action, shape (N, num_actions).
                - value: Scalar value estimate, shape (N, 1).
        """
        # Initial Convolutional Block: Conv -> BN -> ReLU.
        x = F.relu(self.start_bn(self.start_conv(x)))

        # Pass through the stack of Residual Blocks.
        x = self.backbone(x)

        # Policy Head: Conv -> BN -> ReLU -> Flatten -> Linear.
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value Head: Conv -> BN -> ReLU -> Flatten -> Linear -> ReLU -> Linear.
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)

        return policy, value
