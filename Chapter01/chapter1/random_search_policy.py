'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 1: Getting started with reinforcement learning and PyTorch
Author: Yuxi (Hayden) Liu
'''

import gym
import torch

env = gym.make('CartPole-v0')

n_state = env.observation_space.shape[0]
n_action = env.action_space.n


def run_episode(env, weight):
    state = env.reset()
    total_reward = 0
    is_done = False
    while not is_done:
        state = torch.from_numpy(state).float()
        action = torch.argmax(torch.matmul(state, weight))
        state, reward, is_done, _ = env.step(action.item())
        total_reward += reward
    return total_reward



n_episode = 1000

best_total_reward = 0
best_weight = None

total_rewards = []

for episode in range(n_episode):
    weight = torch.rand(n_state, n_action)
    total_reward = run_episode(env, weight)
    print('Episode {}: {}'.format(episode+1, total_reward))
    if total_reward > best_total_reward:
        best_weight = weight
        best_total_reward = total_reward
    total_rewards.append(total_reward)

print('Average total reward over {} episode: {}'.format(n_episode, sum(total_rewards) / n_episode))


import matplotlib.pyplot as plt
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()


n_episode_eval = 100
total_rewards_eval = []
for episode in range(n_episode_eval):
    total_reward = run_episode(env, best_weight)
    print('Episode {}: {}'.format(episode+1, total_reward))
    total_rewards_eval.append(total_reward)


print('Average total reward over {} episode: {}'.format(n_episode, sum(total_rewards_eval) / n_episode_eval))




n_episode = 1000
best_total_reward = 0
best_weight = None
total_rewards = []

for episode in range(n_episode):
    weight = torch.rand(n_state, n_action)
    total_reward = run_episode(env, weight)
    print('Episode {}: {}'.format(episode+1, total_reward))
    if total_reward > best_total_reward:
        best_weight = weight
        best_total_reward = total_reward
    total_rewards.append(total_reward)
    if best_total_reward == 200:
        break



n_training = 1000
n_episode_training = []
for _ in range(n_training):
    for episode in range(n_episode):
        weight = torch.rand(n_state, n_action)
        total_reward = run_episode(env, weight)
        if total_reward == 200:
            n_episode_training.append(episode+1)
            break

print('Expectation of training episodes needed: ', sum(n_episode_training) / n_training)