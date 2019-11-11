'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 8: Implementing Policy Gradients and Policy Optimization
Author: Yuxi (Hayden) Liu
'''

import torch
import gym
import torch.nn as nn
from torch.autograd import Variable

env = gym.make('CartPole-v0')



class Estimator():
    def __init__(self, n_state, lr=0.001):
        self.model = nn.Sequential(
                        nn.Linear(n_state, 1),
                        nn.Sigmoid()
                )
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def predict(self, s):
        return self.model(torch.Tensor(s))

    def update(self, s, y):
        """
        Update the weights of the estimator given the training samples
        """
        y_pred = self.predict(s)
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




def cross_entropy(env, estimator, n_episode, n_samples):
    """
    Cross-entropy algorithm for policy learning
    @param env: Gym environment
    @param estimator: binary estimator
    @param n_episode: number of episodes
    @param n_samples: number of training samples to use
    """

    experience = []
    for episode in range(n_episode):
        rewards = 0
        actions = []
        states = []
        state = env.reset()

        while True:
            action = env.action_space.sample()
            states.append(state)
            actions.append(action)
            next_state, reward, is_done, _ = env.step(action)
            rewards += reward

            if is_done:
                for state, action in zip(states, actions):
                    experience.append((rewards, state, action))
                break

            state = next_state


    experience = sorted(experience, key=lambda x: x[0], reverse=True)
    select_experience = experience[:n_samples]
    train_states = [exp[1] for exp in select_experience]
    train_actions = [exp[2] for exp in select_experience]

    for _  in range(100):
        estimator.update(train_states, train_actions)



n_state = env.observation_space.shape[0]
lr = 0.01
estimator = Estimator(n_state, lr)


n_episode = 5000
n_samples = 10000

cross_entropy(env, estimator, n_episode, n_samples)


n_episode = 100
total_reward_episode = [0] * n_episode
for episode in range(n_episode):
    state = env.reset()
    is_done = False
    while not is_done:
        action = 1 if estimator.predict(state).item() >= 0.5 else 0
        next_state, reward, is_done, _ = env.step(action)
        total_reward_episode[episode] += reward
        state = next_state


import matplotlib.pyplot as plt
plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
