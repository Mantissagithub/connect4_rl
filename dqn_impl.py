import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import gym_connect4

class LinearDeepQLearning(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQLearning, self).__init__()
        
        input_size = np.prod(input_dims)
        
        self.fc1 = nn.Linear(input_size, input_size * 2)
        self.fc2 = nn.Linear(input_size * 2, n_actions)

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        if len(state.shape) > 1:
            state = state.flatten()
        x = F.relu(self.fc1(state))
        actions = self.fc2(x)

        return actions
    

class Agent():
    def __init__(self, input_dims, num_actions, lr, gamma=0.99, epsilon=1.0, eps_dec=1e-5, eps_min=0.01):
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.num_actions)]

        self.Q = LinearDeepQLearning(self.lr, self.num_actions, self.input_dims)


    # epsilon greedy over here
    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            state = torch.tensor(obs, dtype=torch.float).to(self.Q.device)
            if len(state.shape) > 1:
                state = state.flatten()
            actions = self.Q.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


    def learn(self, state, action, reward, next_state):
        self.Q.optimizer.zero_grad()

        self.states = torch.tensor(state, dtype=torch.float).to(self.Q.device)
        self.next_states = torch.tensor(next_state, dtype=torch.float).to(self.Q.device)
        self.rewards = torch.tensor(reward, dtype=torch.float).to(self.Q.device)
        self.actions = torch.tensor(action).to(self.Q.device)

        self.q_pred = self.Q.forward(self.states)[self.actions]

        q_next = self.Q.forward(self.next_states).max()

        q_target = self.rewards + self.gamma * q_next

        loss = self.Q.loss(self.q_pred, q_target).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])
    
    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)
    # plt.show()
    plt.close()

def main():
    from connect_four_gymnasium import ConnectFourEnv
    env = ConnectFourEnv()
    num_games = 10000
    scores = []
    eps_history = []

    agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n, lr=0.00001)

    for i in range(num_games):
        done = False
        score = 0
        obs, info = env.reset()

        while not done:
            action = agent.choose_action(obs)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            agent.learn(obs, action, reward, next_state)
            obs = next_state
        scores.append(score)
        eps_history.append(agent.epsilon)

        if i%100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'Episode {i}, Score: {score}, Average Score: {avg_score}, Epsilon: {agent.epsilon}')
    
    filename = 'dqn_connect4.png'
    x = [i+1 for i in range(num_games)]
    plot_learning_curve(x, scores, eps_history, filename)

if __name__ == "__main__":
    main()



