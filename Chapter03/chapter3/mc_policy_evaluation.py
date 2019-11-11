'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 3: Monte Carlo Methods For Making Numerical Estimations
Author: Yuxi (Hayden) Liu
'''

import torch
import gym

env = gym.make('FrozenLake-v0')



def run_episode(env, policy):
    state = env.reset()
    rewards = []
    states = [state]
    is_done = False
    while not is_done:
        action = policy[state].item()
        state, reward, is_done, info = env.step(action)
        states.append(state)
        rewards.append(reward)
        if is_done:
            break
    states = torch.tensor(states)
    rewards = torch.tensor(rewards)
    return states, rewards


def mc_prediction_first_visit(env, policy, gamma, n_episode):
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    N = torch.zeros(n_state)
    for episode in range(n_episode):
        states_t, rewards_t = run_episode(env, policy)
        return_t = 0
        first_visit = torch.zeros(n_state)
        G = torch.zeros(n_state)
        for state_t, reward_t in zip(reversed(states_t)[1:], reversed(rewards_t)):
            return_t = gamma * return_t + reward_t
            G[state_t] = return_t
            first_visit[state_t] = 1
        for state in range(n_state):
            if first_visit[state] > 0:
                V[state] += G[state]
                N[state] += 1
    for state in range(n_state):
        if N[state] > 0:
            V[state] = V[state] / N[state]
    return V


gamma = 1

n_episode = 10000

optimal_policy = torch.tensor([0., 3., 3., 3., 0., 3., 2., 3., 3., 1., 0., 3., 3., 2., 1., 3.])

value = mc_prediction_first_visit(env, optimal_policy, gamma, n_episode)

print('The value function calculated by first-visit MC prediction:\n', value)



def mc_prediction_every_visit(env, policy, gamma, n_episode):
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    N = torch.zeros(n_state)
    G = torch.zeros(n_state)
    for episode in range(n_episode):
        states_t, rewards_t = run_episode(env, policy)
        return_t = 0
        for state_t, reward_t in zip(reversed(states_t)[1:], reversed(rewards_t)):
            return_t = gamma * return_t + reward_t
            G[state_t] += return_t
            N[state_t] += 1
    for state in range(n_state):
        if N[state] > 0:
            V[state] = G[state] / N[state]
    return V

value = mc_prediction_every_visit(env, optimal_policy, gamma, n_episode)

print('The value function calculated by every-visit MC prediction:\n', value)

