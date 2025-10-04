# Connect4 RL

Creating a Connect4 agent using Deep Q-Network (DQN) reinforcement learning. This implementation explores how deep neural networks can learn to play Connect4 through Q-learning with experience replay and target networks.

Main Inspiration:
This project implements the foundational Deep Q-Network algorithm, building upon the breakthrough work from DeepMind that first demonstrated how deep learning could be applied to reinforcement learning for game-playing agents.

**Research Foundation:**
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) - Original DQN paper by Mnih et al.
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) - Double DQN improvements
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) - Enhanced replay buffer methods

## Architecture

1. **Game Environment** - Connect4 game logic, state representation, reward system
2. **Deep Q-Network** - Fully connected neural network that estimates Q-values for each action
3. **Experience Replay** - Buffer storing past experiences for batch learning and stability
4. **Target Network** - Separate network for stable Q-value targets during training
5. **Epsilon-Greedy Exploration** - Balanced exploration vs exploitation strategy

## Training Results

**Status:** COMPLETED ✅ - Two DQN implementations

**Implementations:**
1. **Basic DQN** (`dqn_impl.py`) - Simple Q-learning with neural network
2. **Advanced DQN** (`dqn_replay_buffer_target_network.py`) - Experience replay + target network

**Training Configuration:** 10,000 episodes per implementation

| Implementation | Features | Learning Rate | Key Components |
|---------------|----------|---------------|----------------|
| Basic DQN | Epsilon-greedy, Direct learning | 0.00001 | Single network, immediate updates |
| Advanced DQN | Replay buffer, Target network | 0.0001 | Double network, batch learning |

## Results Analysis

The training plots show the learning progression across 10,000 episodes:

- **Basic DQN** (`dqn_connect4.png`): Shows steady improvement with some instability
- **Advanced DQN** (`dqn_replay_target_connect4.png`): More stable learning with replay buffer and target network

## Hardware Specifications

**Training Hardware:**
- **GPU:** NVIDIA RTX 4060 (8GB VRAM)
- **RAM:** 16GB
- **Platform:** Laptop

The RTX 4060's 8GB VRAM proved sufficient for training this Connect4 agent, handling the neural network computations and MCTS simulations efficiently. The 16GB system RAM provided adequate buffer space for training data management and self-play game generation.

## Hardware Specifications

**Training Hardware:**
- **GPU:** NVIDIA RTX 4060 (8GB VRAM)
- **RAM:** 16GB
- **Platform:** Laptop

The RTX 4060's 8GB VRAM was more than sufficient for training these DQN agents, as the neural networks are relatively small compared to modern deep learning models. The fully connected architecture requires minimal GPU memory, making this an accessible implementation for learning DQN concepts.

## Key Learning Outcomes

This implementation demonstrates core DQN concepts:

1. **Q-Learning with Neural Networks**: How deep networks can approximate Q-values for complex state spaces
2. **Experience Replay**: The importance of decorrelated training data for stable learning
3. **Target Networks**: How separate target networks prevent moving target problems
4. **Exploration vs Exploitation**: Epsilon-greedy strategy balancing learning and performance

The comparison between basic and advanced DQN implementations clearly shows the benefits of experience replay and target networks for training stability and convergence.

*This project provides a practical understanding of how DQN revolutionized reinforcement learning by combining Q-learning with deep neural networks, making it possible to tackle problems with high-dimensional state spaces.*
