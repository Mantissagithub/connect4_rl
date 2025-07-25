# Connect4 RL

Creating a self-playing Connect4 agent using reinforcement learning, inspired by AlphaGo/AlphaZero algorithms. Built in 8 hours to understand how policy networks and value networks combine with Monte Carlo Tree Search for strategic gameplay.

Main Inspiration:
This project was deeply inspired by the groundbreaking [AlphaGo documentary (youtube)](https://youtu.be/WXuK6gekU1Y?si=ZPMd_DLDlVjDA63T) from 5 years ago, which showcased the historic match between Lee Sedol and DeepMind's AlphaGo in 2016. The documentary beautifully captures the moment when artificial intelligence achieved a milestone that many thought was decades away - mastering the ancient game of Go through pure self-play and neural networks.

**Research Foundation:**
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) - Original AlphaZero paper
- [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://www.science.org/doi/10.1126/science.aar6404) - Nature publication
- 

## Architecture

1. **Game Engine** - Board management, move validation, win detection, state conversion
2. **Neural Network** - Convolutional feature extraction with policy head (move probabilities) and value head (position evaluation)
3. **MCTS** - Tree search using UCB scoring, expansion, neural network evaluation, and backpropagation
4. **Training Data** - Self-play game generation, experience collection, replay buffer management
5. **Training Loop** - Iterative data generation and network updates with checkpointing

## Training Results

**Status:** COMPLETED ✅ in 16.80 hours (1008 minutes)

**Iterations:** 1000 | **Best Rating:** 14.00 

**Model:** 97,047 parameters | **Checkpoint:** `checkpoints/checkpoint_iter_1000_20250713_192301.pt`

| Metric | Value |
|--------|-------|
| Training Duration |16.80 hours |
| Final Iteration | 1000 |
| Best Rating | 14.00 |
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
