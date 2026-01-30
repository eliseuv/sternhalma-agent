# General

- [ ] Implement some kind of potential function to guide the agent to the best moves

# Testing

Avoid silent failures where the program runs but the agent never learns because the math isn't right

- [ ] Print the loss at each step
- [ ] Replay buffer: make sure it stores and samples the transitions correctly
- [ ] Make sure the weights are copied from the target network to the evaluation network every few steps
- [ ] Ensure the board state inputs are normalized instead of raw arbitrary IDs
