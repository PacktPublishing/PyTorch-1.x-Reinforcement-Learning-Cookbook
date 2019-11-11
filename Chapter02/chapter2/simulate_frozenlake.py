'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 2: Markov Decision Process and Dynamic Programming
Author: Yuxi (Hayden) Liu
'''

import gym
import torch


env = gym.make("FrozenLake-v0")

n_state = env.observation_space.n
print(n_state)
n_action = env.action_space.n
print(n_action)


env.reset()

env.render()

new_state, reward, is_done, info = env.step(1)
env.render()
print(new_state)
print(reward)
print(is_done)
print(info)




def run_episode(env, policy):
    state = env.reset()
    total_reward = 0
    is_done = False
    while not is_done:
        action = policy[state].item()
        state, reward, is_done, info = env.step(action)
        total_reward += reward
        if is_done:
            break
    return total_reward



n_episode = 1000

total_rewards = []
for episode in range(n_episode):
    random_policy = torch.randint(high=n_action, size=(n_state,))
    total_reward = run_episode(env, random_policy)
    total_rewards.append(total_reward)

print('Average total reward under random policy: {}'.format(sum(total_rewards) / n_episode))




while True:
    random_policy = torch.randint(high=n_action, size=(n_state,))
    total_reward = run_episode(env, random_policy)
    if total_reward == 1:
        best_policy = random_policy
        break

total_rewards = []
for episode in range(n_episode):
    total_reward = run_episode(env, best_policy)
    total_rewards.append(total_reward)

print(best_policy)
print('Average total reward under random search policy: {}'.format(sum(total_rewards) / n_episode))



print(env.env.P[6])
print(env.env.P[11])