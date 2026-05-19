# Connect4 RL Training Math

This document defines the mathematical objects and update rules used by the training pipeline in this repository. The implementation is AlphaZero-style in spirit: self-play generates policy and value targets, Monte Carlo Tree Search improves action selection, and a shared policy-value network is trained on the resulting dataset.

## 1. Game Model

Connect4 is treated as a two-player, deterministic, zero-sum, alternating-turn game.

- State space: \(s \in \mathcal{S}\)
- Action space: \(a \in \mathcal{A} = \{0,1,2,3,4,5,6\}\)
- Transition function: \(s' = T(s, a)\)
- Terminal outcome: \(z \in \{-1, 0, 1\}\)

Outcome semantics:

- \(z = 1\): win for the player whose perspective defines the sample
- \(z = 0\): draw
- \(z = -1\): loss

## 2. State Encoding

Each board position is encoded as a tensor of shape:

$$
3 \times 6 \times 7
$$

using current-player perspective:

- channel 0: current player discs
- channel 1: opponent discs
- channel 2: empty cells

This means the network is always asked to evaluate from the viewpoint of the player to move in that position.

## 3. Policy-Value Network

The model represents a function:

$$
f_\theta(s) = \left(p_\theta(\cdot \mid s), v_\theta(s)\right)
$$

where:

- \(p_\theta(\cdot \mid s) \in \mathbb{R}^7\) are policy logits over columns
- \(v_\theta(s) \in [-1,1]\) is the scalar value estimate

After softmax:

$$
\pi_\theta(a \mid s) = \frac{\exp(p_\theta(a \mid s))}{\sum_b \exp(p_\theta(b \mid s))}
$$

The value head uses a final `tanh`, so:

$$
v_\theta(s) \approx \mathbb{E}[z \mid s]
$$

from the perspective of the player to move.

## 4. Monte Carlo Tree Search

For a search node \(s\), each legal action \(a\) maintains:

- prior \(P(s,a)\)
- visit count \(N(s,a)\)
- cumulative value \(W(s,a)\)
- mean value \(Q(s,a)\)

with:

$$
Q(s,a) = \frac{W(s,a)}{N(s,a)}
$$

for visited actions.

### 4.1 Selection

Child selection uses a prior-guided UCB score:

$$
\mathrm{UCB}(s,a) = Q(s,a) + c_{\text{puct}} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}
$$

where:

- \(N(s)\) is the parent visit count
- \(c_{\text{puct}}\) is the exploration constant

### 4.2 Expansion

When a leaf is expanded:

1. the environment generates all legal successor states
2. the network produces policy logits and value
3. policy probabilities are masked to legal actions and renormalized
4. those masked probabilities become child priors

### 4.3 Backpropagation

If the leaf value is \(v\), the path value is propagated upward with alternating sign:

$$
v,\; -v,\; v,\; -v,\; \dots
$$

This sign flip is required because players alternate turns. A position that is good for the current player is bad for the opponent one ply above.

## 5. Self-Play Targets

For each self-play move, the training sample stores:

- encoded state \(s_t\)
- MCTS-improved policy target \(\pi_t\)
- player-to-move label
- final value target \(z_t\)

### 5.1 Policy Target

The search policy target is computed from root visit counts:

$$
\pi_t(a) = \frac{N(s_t, a)}{\sum_b N(s_t, b)}
$$

If no usable visits are available, the code falls back to a masked network prior or uniform legal-action distribution.

### 5.2 Value Target

Let the final winner be \(w \in \{0,1,2\}\). For a sample generated when player \(p\) was to move:

$$
z_t =
\begin{cases}
0 & \text{if } w = 0 \\
1 & \text{if } w = p \\
-1 & \text{if } w \neq p
\end{cases}
$$

So every sample is labeled from its own acting player's perspective.

## 6. Loss Function

The network is optimized with a sum of policy loss and value loss:

$$
\mathcal{L} = \lambda_\pi \mathcal{L}_{\pi} + \lambda_v \mathcal{L}_v
$$

with default weights:

$$
\lambda_\pi = 1,\quad \lambda_v = 1
$$

### 6.1 Policy Loss

The repository uses KL divergence between the MCTS target distribution and the model policy:

$$
\mathcal{L}_{\pi}
=
\mathrm{KL}\left(\pi_{\text{MCTS}} \,\|\, \pi_\theta\right)
$$

implemented through log-softmax on policy logits.

### 6.2 Value Loss

The value objective is mean squared error:

$$
\mathcal{L}_v = \left(v_\theta(s) - z\right)^2
$$

batched as standard MSE over the sampled replay batch.

## 7. Replay Buffer and Batch Sampling

Self-play examples are accumulated in a replay buffer of bounded size.

If the buffer contains samples:

$$
\mathcal{D} = \{(s_i, \pi_i, z_i)\}_{i=1}^N
$$

then each training epoch samples mini-batches:

$$
\mathcal{B} \subset \mathcal{D}, \quad |\mathcal{B}| = B
$$

where \(B\) is the configured batch size.

The current implementation samples uniformly from the buffer for each batch.

## 8. Optimization

Parameters are updated by Adam:

$$
\theta \leftarrow \mathrm{Adam}\left(\theta, \nabla_\theta \mathcal{L}\right)
$$

with:

- initial learning rate: \(10^{-3}\)
- weight decay: \(10^{-4}\)
- gradient clipping: max norm \(= 1.0\)

The main loop also applies a step scheduler:

$$
\eta_k = \eta_0 \cdot \gamma^{\left\lfloor k / s \right\rfloor}
$$

where:

- \(\eta_0\) is the initial learning rate
- \(s = 100\) iterations
- \(\gamma = 0.9\)

## 9. Evaluation Score

The repository's internal score is not Elo. It is a custom scalar:

$$
\mathrm{score}
=
100 \cdot \mathrm{win\_rate\_vs\_random}
+
20 \cdot (1 - \mathrm{win\_rate\_balance})
$$

where:

$$
\mathrm{win\_rate\_balance}
=
\frac{|\mathrm{P1Wins} - \mathrm{P2Wins}|}{\mathrm{GamesPlayed}}
$$

This gives a nominal range:

$$
0 \le \mathrm{score} \le 120
$$

Higher values mean:

- stronger play against a random baseline
- less first-player/second-player asymmetry in self-play

## 10. What “Learning” Looks Like

In a healthy run, the following usually happen together:

- replay buffer size grows steadily
- policy and value losses are finite and nonzero
- epoch summaries vary over time instead of staying at exactly zero
- benchmark score improves or stabilizes at a high level across checkpoints

Signs of failure include:

- `final_loss = 0.0` for many iterations
- repeated self-play collapse into degenerate, ultra-short games
- evaluation score that never changes
- exceptions inside training that prevent gradient updates

## 11. Mapping to the Code

Main implementation locations:

- environment and state encoding: `game_engine_components/`
- search: `mcts_components/`
- self-play target generation: `training_data_components/generate_self_play_game.py`
- optimization loop: `training_loop_components/network_training.py`
- outer training driver: `main_training_loop.py`

This file is the mathematical companion to the implementation, not a separate algorithm spec. If behavior and this note diverge, the code is the source of truth.
