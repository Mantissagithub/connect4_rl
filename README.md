# Connect4 RL

Creating a self-playing Connect4 agent using reinforcement learning, inspired by AlphaGo/AlphaZero algorithms. Built in 8 hours to understand how policy networks and value networks combine with Monte Carlo Tree Search for strategic gameplay.

**Research Foundation:**
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) - Original AlphaZero paper
- [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://www.science.org/doi/10.1126/science.aar6404) - Nature publication

## Architecture

1. **Game Engine** - Board management, move validation, win detection, state conversion
2. **Neural Network** - Convolutional feature extraction with policy head (move probabilities) and value head (position evaluation)
3. **MCTS** - Tree search using UCB scoring, expansion, neural network evaluation, and backpropagation
4. **Training Data** - Self-play game generation, experience collection, replay buffer management
5. **Training Loop** - Iterative data generation and network updates with checkpointing

## Training Results

**Status:** COMPLETED ✅ in 0.57 hours (34 minutes)

**Iterations:** 500 | **Best Rating:** 4.80 | **Peak Score:** 12.08

**Model:** 97,047 parameters | **Checkpoint:** `checkpoint_iter_0500_20250712_162910.pt`

| Metric | Value |
|--------|-------|
| Training Duration | 0.57 hours |
| Final Iteration | 500 |
| Best Rating | 4.80 |
| Peak Score | 12.08 |
| Model Size | 97,047 parameters |

## Hardware Specifications

**Training Hardware:**
- **GPU:** NVIDIA RTX 4060 (8GB VRAM)
- **RAM:** 16GB
- **Platform:** Laptop

The RTX 4060's 8GB VRAM proved sufficient for training this Connect4 agent, handling the neural network computations and MCTS simulations efficiently. The 16GB system RAM provided adequate buffer space for training data management and self-play game generation.

Fascinating to watch an AI learn strategic thinking from scratch through pure self-play - the agent progressed from random moves to genuine strategic understanding across 500 training iterations. The policy and value networks working together with MCTS created a beautiful feedback loop that gradually produced stronger gameplay.

*This implementation captures core AlphaZero concepts with modest computational resources, proving the fundamental ideas work even without DeepMind's massive TPU clusters.*

If time permits, will extend training with more iterations to push the score higher and see how far this agent can evolve its strategic understanding.
