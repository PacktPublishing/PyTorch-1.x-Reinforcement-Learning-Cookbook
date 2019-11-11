'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 3: Monte Carlo Methods For Making Numerical Estimations
Author: Yuxi (Hayden) Liu
'''

import torch
import gym

env = gym.make('Blackjack-v0')


def gen_random_policy(n_action):
    probs = torch.ones(n_action) / n_action
    def policy_function(state):
        return probs
    return policy_function


def run_episode(env, behavior_policy):
    """
    Run a episode given a behavior policy
    @param env: OpenAI Gym environment
    @param behavior_policy: behavior policy
    @return: resulting states, actions and rewards for the entire episode
    """
    state = env.reset()
    rewards = []
    actions = []
    states = []
    is_done = False
    while not is_done:
        probs = behavior_policy(state)
        action = torch.multinomial(probs, 1).item()
        actions.append(action)
        states.append(state)
        state, reward, is_done, info = env.step(action)
        rewards.append(reward)
        if is_done:
            break
    return states, actions, rewards



from collections import defaultdict


def mc_control_off_policy_weighted(env, gamma, n_episode, behavior_policy):
    """
    Obtain the optimal policy with off-policy MC control method with weighted importance sampling
    @param env: OpenAI Gym environment
    @param gamma: discount factor
    @param n_episode: number of episodes
    @param behavior_policy: behavior policy
    @return: the optimal Q-function, and the optimal policy
    """
    n_action = env.action_space.n
    N = defaultdict(float)
    Q = defaultdict(lambda: torch.empty(n_action))
    for episode in range(n_episode):
        W = 1.
        states_t, actions_t, rewards_t = run_episode(env, behavior_policy)
        return_t = 0.
        for state_t, action_t, reward_t in zip(states_t[::-1], actions_t[::-1], rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            N[(state_t, action_t)] += W
            Q[state_t][action_t] += (W / N[(state_t, action_t)]) * (return_t - Q[state_t][action_t])
            if action_t != torch.argmax(Q[state_t]).item():
                break
            W *= 1./ behavior_policy(state_t)[action_t]
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy

gamma = 1

n_episode = 500000

random_policy = gen_random_policy(env.action_space.n)

optimal_Q, optimal_policy = mc_control_off_policy_weighted(env, gamma, n_episode, random_policy)




def simulate_episode(env, policy):
    state = env.reset()
    is_done = False
    while not is_done:
        action = policy[state]
        state, reward, is_done, info = env.step(action)
        if is_done:
            return reward


n_episode = 100000
n_win_optimal = 0
n_lose_optimal = 0

for _ in range(n_episode):
    reward = simulate_episode(env, optimal_policy)
    if reward == 1:
        n_win_optimal += 1
    elif reward == -1:
        n_lose_optimal += 1



print('Winning probability under the optimal policy: {}'.format(n_win_optimal/n_episode))

print('Losing probability under the optimal policy: {}'.format(n_lose_optimal/n_episode))

