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
    total_reward = 0
    is_done = False
    while not is_done:
        state = torch.from_numpy(state).float()
        action = torch.argmax(torch.matmul(state, weight))
        state, reward, is_done, _ = env.step(action.item())
        total_reward += reward
    return total_reward


n_episode = 1000
best_weight = torch.rand(n_state, n_action)
best_total_reward = 0
total_rewards = []

noise_scale = 0.01

for episode in range(n_episode):
    weight = best_weight + noise_scale * torch.rand(n_state, n_action)
    total_reward = run_episode(env, weight)
    if total_reward >= best_total_reward:
        best_total_reward = total_reward
        best_weight = weight
    total_rewards.append(total_reward)
    print('Episode {}: {}'.format(episode + 1, total_reward))

print('Average total reward over {} episode: {}'.format(n_episode, sum(total_rewards) / n_episode))



best_weight = torch.rand(n_state, n_action)
noise_scale = 0.01

best_total_reward = 0
total_rewards = []
for episode in range(n_episode):
    weight = best_weight + noise_scale * torch.rand(n_state, n_action)
    total_reward = run_episode(env, weight)
    if total_reward >= best_total_reward:
        best_total_reward = total_reward
        best_weight = weight
        noise_scale = max(noise_scale / 2, 1e-4)
    else:
        noise_scale = min(noise_scale * 2, 2)

    print('Episode {}: {}'.format(episode + 1, total_reward))
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





best_weight = torch.rand(n_state, n_action)
noise_scale = 0.01

best_total_reward = 0
total_rewards = []
for episode in range(n_episode):
    weight = best_weight + noise_scale * torch.rand(n_state, n_action)
    total_reward = run_episode(env, weight)
    if total_reward >= best_total_reward:
        best_total_reward = total_reward
        best_weight = weight
        noise_scale = max(noise_scale / 2, 1e-4)
    else:
        noise_scale = min(noise_scale * 2, 2)
    print('Episode {}: {}'.format(episode + 1, total_reward))
    total_rewards.append(total_reward)
    if episode >= 99 and sum(total_rewards[-100:]) >= 19500:
        break


