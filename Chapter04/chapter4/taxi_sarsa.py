'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 4: Temporal Difference and Q-Learning
Author: Yuxi (Hayden) Liu
'''

import torch
import gym

env = gym.make('Taxi-v2')


def gen_epsilon_greedy_policy(n_action, epsilon):
    def policy_function(state, Q):
        probs = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function


from collections import defaultdict

def sarsa(env, gamma, n_episode, alpha):
    """
    Obtain the optimal policy with on-policy SARSA algorithm
    @param env: OpenAI Gym environment
    @param gamma: discount factor
    @param n_episode: number of episodes
    @return: the optimal Q-function, and the optimal policy
    """
    n_action = env.action_space.n
    Q = defaultdict(lambda: torch.zeros(n_action))
    for episode in range(n_episode):
        state = env.reset()
        is_done = False
        action = epsilon_greedy_policy(state, Q)
        while not is_done:
            next_state, reward, is_done, info = env.step(action)
            next_action = epsilon_greedy_policy(next_state, Q)
            td_delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            Q[state][action] += alpha * td_delta
            length_episode[episode] += 1
            total_reward_episode[episode] += reward
            if is_done:
                break
            state = next_state
            action = next_action
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy

gamma = 1

n_episode = 1000


alpha = 0.4

epsilon = 0.01

epsilon_greedy_policy = gen_epsilon_greedy_policy(env.action_space.n, epsilon)

length_episode = [0] * n_episode
total_reward_episode = [0] * n_episode

optimal_Q, optimal_policy = sarsa(env, gamma, n_episode, alpha)

import matplotlib.pyplot as plt
plt.plot(length_episode)
plt.title('Episode length over time')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.show()


plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()



alpha_options = [0.4, 0.5, 0.6]
epsilon_options = [0.1, 0.03, 0.01]
n_episode = 500

for alpha in alpha_options:
    for epsilon in epsilon_options:
        length_episode = [0] * n_episode
        total_reward_episode = [0] * n_episode
        sarsa(env, gamma, n_episode, alpha)
        reward_per_step = [reward/float(step) for reward, step in zip(total_reward_episode, length_episode)]
        print('alpha: {}, epsilon: {}'.format(alpha, epsilon))
        print('Average reward over {} episodes: {}'.format(n_episode, sum(total_reward_episode) / n_episode))
        print('Average length over {} episodes: {}'.format(n_episode, sum(length_episode) / n_episode))
        print('Average reward per step over {} episodes: {}\n'.format(n_episode, sum(reward_per_step) / n_episode))




