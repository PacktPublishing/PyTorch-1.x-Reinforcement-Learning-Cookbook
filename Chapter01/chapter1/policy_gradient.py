'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 1: Getting started with reinforcement learning and PyTorch
Author: Yuxi (Hayden) Liu
'''

import torch
import gym


env = gym.make('CartPole-v0')

n_state = env.observation_space.shape[0]
n_action = env.action_space.n


def run_episode(env, weight):
    state = env.reset()
    grads = []
    total_reward = 0
    is_done = False
    while not is_done:
        state = torch.from_numpy(state).float()
        z = torch.matmul(state, weight)
        probs = torch.nn.Softmax()(z)
        action = int(torch.bernoulli(probs[1]).item())
        d_softmax = torch.diag(probs) - probs.view(-1, 1) * probs
        d_log = d_softmax[action] / probs[action]
        grad = state.view(-1, 1) * d_log
        grads.append(grad)
        state, reward, is_done, _ = env.step(action)
        total_reward += reward
        if is_done:
            break
    return total_reward, grads



n_episode = 1000
learning_rate = 0.001

total_rewards = []

weight = torch.rand(n_state, n_action)

for episode in range(n_episode):
    total_reward, gradients = run_episode(env, weight)
    print('Episode {}: {}'.format(episode + 1, total_reward))
    for i, gradient in enumerate(gradients):
        weight += learning_rate * gradient * (total_reward - i)
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
    total_reward, _ = run_episode(env, weight)
    print('Episode {}: {}'.format(episode+1, total_reward))
    total_rewards_eval.append(total_reward)


print('Average total reward over {} episode: {}'.format(n_episode, sum(total_rewards_eval) / n_episode_eval))
