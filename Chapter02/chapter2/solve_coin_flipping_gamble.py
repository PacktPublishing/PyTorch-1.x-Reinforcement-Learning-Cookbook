'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 2: Markov Decision Process and Dynamic Programming
Author: Yuxi (Hayden) Liu
'''

import torch

capital_max = 100
n_state = capital_max + 1
rewards = torch.zeros(n_state)
rewards[-1] = 1

print(rewards)

gamma = 1
threshold = 1e-10

head_prob = 0.4

env = {'capital_max': capital_max,
       'head_prob': head_prob,
       'rewards': rewards,
       'n_state': n_state}


def value_iteration(env, gamma, threshold):
    """
    Solve the coin flipping gamble problem with value iteration algorithm
    @param env: the coin flipping gamble environment
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: values of the optimal policy for the given environment
    """
    head_prob = env['head_prob']
    n_state = env['n_state']
    capital_max = env['capital_max']
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state in range(1, capital_max):
            v_actions = torch.zeros(min(state, capital_max - state) + 1)
            for action in range(1, min(state, capital_max - state) + 1):
                v_actions[action] += head_prob * (rewards[state + action] + gamma * V[state + action])
                v_actions[action] += (1 - head_prob) * (rewards[state - action] + gamma * V[state - action])
            V_temp[state] = torch.max(v_actions)
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if max_delta <= threshold:
            break
    return V


def extract_optimal_policy(env, V_optimal, gamma):
    """
    Obtain the optimal policy based on the optimal values
    @param env: the coin flipping gamble environment
    @param V_optimal: optimal values
    @param gamma: discount factor
    @return: optimal policy
    """
    head_prob = env['head_prob']
    n_state = env['n_state']
    capital_max = env['capital_max']
    optimal_policy = torch.zeros(capital_max).int()
    for state in range(1, capital_max):
        v_actions = torch.zeros(n_state)
        for action in range(1, min(state, capital_max - state) + 1):
            v_actions[action] += head_prob * (rewards[state + action] + gamma * V_optimal[state + action])
            v_actions[action] += (1 - head_prob) * (rewards[state - action] + gamma * V_optimal[state - action])
        optimal_policy[state] = torch.argmax(v_actions)
    return optimal_policy


import time

start_time = time.time()
V_optimal = value_iteration(env, gamma, threshold)
optimal_policy = extract_optimal_policy(env, V_optimal, gamma)

print("It takes {:.3f}s to solve with value iteration".format(time.time() - start_time))

print('Optimal values:\n{}'.format(V_optimal))
print('Optimal policy:\n{}'.format(optimal_policy))

import matplotlib.pyplot as plt

plt.plot(V_optimal[:100].numpy())
plt.title('Optimal policy values')
plt.xlabel('Capital')
plt.ylabel('Policy value')
plt.show()


#
# plt.bar(range(1, capital_max), optimal_policy[1:capital_max].numpy())
# plt.title('Optimal policy')
# plt.xlabel('Capital')
# plt.ylabel('Optimal action')
# plt.show()






def policy_evaluation(env, policy, gamma, threshold):
    """
    Perform policy evaluation
    @param env: the coin flipping gamble environment
    @param policy: policy tensor containing actions taken for individual state
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: values of the given policy
    """
    head_prob = env['head_prob']
    n_state = env['n_state']
    capital_max = env['capital_max']
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state in range(1, capital_max):
            action = policy[state].item()
            V_temp[state] += head_prob * (rewards[state + action] + gamma * V[state + action])
            V_temp[state] += (1 - head_prob) * (rewards[state - action] + gamma * V[state - action])
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if max_delta <= threshold:
            break
    return V


def policy_improvement(env, V, gamma):
    """
    Obtain an improved policy based on the values
    @param env: the coin flipping gamble environment
    @param V: policy values
    @param gamma: discount factor
    @return: the policy
    """
    head_prob = env['head_prob']
    n_state = env['n_state']
    capital_max = env['capital_max']
    policy = torch.zeros(n_state).int()
    for state in range(1, capital_max):
        v_actions = torch.zeros(min(state, capital_max - state) + 1)
        for action in range(1, min(state, capital_max - state) + 1):
            v_actions[action] += head_prob * (rewards[state + action] + gamma * V[state + action])
            v_actions[action] += (1 - head_prob) * (rewards[state - action] + gamma * V[state - action])
        policy[state] = torch.argmax(v_actions)
    return policy


def policy_iteration(env, gamma, threshold):
    """
    Solve the coin flipping gamble problem with policy iteration algorithm
    @param env: the coin flipping gamble environment
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: optimal values and the optimal policy for the given environment
    """
    n_state = env['n_state']
    policy = torch.zeros(n_state).int()
    while True:
        V = policy_evaluation(env, policy, gamma, threshold)
        policy_improved = policy_improvement(env, V, gamma)
        if torch.equal(policy_improved, policy):
            return V, policy_improved
        policy = policy_improved


start_time = time.time()
V_optimal, optimal_policy = policy_iteration(env, gamma, threshold)

print("It takes {:.3f}s to solve with policy iteration".format(time.time() - start_time))

print('Optimal values:\n{}'.format(V_optimal))
print('Optimal policy:\n{}'.format(optimal_policy))

exit()


def run_random_episode(head, capital):
    while capital > 0:
        # print(capital)
        # bet = torch.randint(1, capital + 1, (1,)).item()
        bet = 1
        if torch.rand(1).item() < head:
            capital += bet
            if capital >= 100:
                return 1
        else:
            capital -= bet
    return 0


def run_optimal_episode(head, capital, optimal_policy):
    while capital > 0:
        bet = optimal_policy[capital].item()
        if torch.rand(1).item() < head:
            capital += bet
            if capital >= 100:
                return 1
        else:
            capital -= bet
    return 0


capital = 50

n_episode = 5000
total_rewards_random = []
total_rewards_opt = []
for episode in range(n_episode):
    total_reward_random = run_random_episode(0.48, capital)
    total_reward_opt = run_optimal_episode(0.4, capital, optimal_policy)
    total_rewards_random.append(total_reward_random)
    total_rewards_opt.append(total_reward_opt)

print('Average total reward under the optimal policy: {}'.format(sum(total_rewards_random) / n_episode))
print('Average total reward under the optimal policy: {}'.format(sum(total_rewards_opt) / n_episode))
