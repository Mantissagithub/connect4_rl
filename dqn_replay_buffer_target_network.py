import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import torch
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import gym_connect4
from collections import deque
import random

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
        if len(state.shape) == 1:
            pass
        elif len(state.shape) == 2 and state.shape[0] > 1:
            pass
        elif len(state.shape) > 2:
            state = state.flatten()
        
        x = F.relu(self.fc1(state))
        actions = self.fc2(x)
        return actions

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class Agent():
    def __init__(self, input_dims, num_actions, lr, gamma=0.99, epsilon=1.0, eps_dec=1e-5, eps_min=0.01, 
                 buffer_size=100000, batch_size=64, target_update=100):
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.learn_step_counter = 0
        self.action_space = [i for i in range(self.num_actions)]

        self.Q = LinearDeepQLearning(self.lr, self.num_actions, self.input_dims)
        self.Q_target = LinearDeepQLearning(self.lr, self.num_actions, self.input_dims)
        
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_target.eval()
        
        self.replay_buffer = ReplayBuffer(buffer_size)

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

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states_array = np.array([np.array(state).flatten() for state in states])
        next_states_array = np.array([np.array(state).flatten() for state in next_states])
        
        states = torch.tensor(states_array, dtype=torch.float).to(self.Q.device)
        actions = torch.tensor(actions).to(self.Q.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.Q.device)
        next_states = torch.tensor(next_states_array, dtype=torch.float).to(self.Q.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.Q.device)
        
        current_q_values = self.Q(states).gather(1, actions.unsqueeze(1))
        
        next_q_values = self.Q_target(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = self.Q.loss(current_q_values.squeeze(), target_q_values)
        
        self.Q.optimizer.zero_grad()
        loss.backward()
        self.Q.optimizer.step()
        
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
        
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
    plt.close()

def main():
    from connect_four_gymnasium import ConnectFourEnv
    env = ConnectFourEnv()
    num_games = 10000
    scores = []
    eps_history = []

    agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n, lr=0.0001)

    for i in range(num_games):
        done = False
        score = 0
        obs, info = env.reset()

        while not done:
            action = agent.choose_action(obs)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            
            agent.store_transition(obs, action, reward, next_state, done)
            
            agent.learn()
            
            obs = next_state
            
        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'Episode {i}, Score: {score}, Average Score: {avg_score}, Epsilon: {agent.epsilon:.3f}')
    
    filename = 'dqn_replay_target_connect4.png'
    x = [i+1 for i in range(num_games)]
    plot_learning_curve(x, scores, eps_history, filename)

if __name__ == "__main__":
    main()



