'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 3: Monte Carlo Methods For Making Numerical Estimations
Author: Yuxi (Hayden) Liu
'''

import torch
import gym

env = gym.make('Blackjack-v0')



def run_episode(env, Q, n_action):
    """
    Run a episode given a Q-function
    @param env: OpenAI Gym environment
    @param Q: Q-function
    @param n_action: action space
    @return: resulting states, actions and rewards for the entire episode
    """
    state = env.reset()
    rewards = []
    actions = []
    states = []
    is_done = False
    action = torch.randint(0, n_action, [1]).item()
    while not is_done:
        actions.append(action)
        states.append(state)
        state, reward, is_done, info = env.step(action)
        rewards.append(reward)
        if is_done:
            break
        action = torch.argmax(Q[state]).item()
    return states, actions, rewards



from collections import defaultdict

def mc_control_on_policy(env, gamma, n_episode):
    """
    Obtain the optimal policy with on-policy MC control method
    @param env: OpenAI Gym environment
    @param gamma: discount factor
    @param n_episode: number of episodes
    @return: the optimal Q-function, and the optimal policy
    """
    n_action = env.action_space.n
    G_sum = defaultdict(float)
    N = defaultdict(int)
    Q = defaultdict(lambda: torch.empty(n_action))
    for episode in range(n_episode):
        states_t, actions_t, rewards_t = run_episode(env, Q, n_action)
        return_t = 0
        G = {}
        for state_t, action_t, reward_t in zip(states_t[::-1], actions_t[::-1], rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            G[(state_t, action_t)] = return_t
        for state_action, return_t in G.items():
            state, action = state_action
            if state[0] <= 21:
                G_sum[state_action] += return_t
                N[state_action] += 1
                Q[state][action] = G_sum[state_action] / N[state_action]
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy


gamma = 1

n_episode = 500000

optimal_Q, optimal_policy = mc_control_on_policy(env, gamma, n_episode)

print(optimal_policy)

optimal_value = defaultdict(float)
for state, action_values in optimal_Q.items():
    optimal_value[state] = torch.max(action_values).item()


print('The value function of the optimal policy:\n', optimal_value)





import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_surface(X, Y, Z, title):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Showing')
    ax.set_zlabel('Value')
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    fig.colorbar(surf)
    plt.show()


def plot_blackjack_value(V):
    player_sum_range = range(12, 22)
    dealer_show_range = range(1, 11)
    X, Y = torch.meshgrid([torch.tensor(player_sum_range), torch.tensor(dealer_show_range)])
    values_to_plot = torch.zeros((len(player_sum_range), len(dealer_show_range), 2))
    for i, player in enumerate(player_sum_range):
        for j, dealer in enumerate(dealer_show_range):
            for k, ace in enumerate([False, True]):
                values_to_plot[i, j, k] = V[(player, dealer, ace)]
    plot_surface(X, Y, values_to_plot[:,:,0].numpy(), "Blackjack Value Function Without Usable Ace")
    plot_surface(X, Y, values_to_plot[:,:,1].numpy(), "Blackjack Value Function With Usable Ace")


plot_blackjack_value(optimal_value)





hold_score = 18

hold_policy = {}
player_sum_range = range(2, 22)
for player in range(2, 22):
    for dealer in range(1, 11):
        action = 1 if player < hold_score else 0
        hold_policy[(player, dealer, False)] = action
        hold_policy[(player, dealer, True)] = action



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
n_win_simple = 0
n_lose_optimal = 0
n_lose_simple = 0

for _ in range(n_episode):
    reward = simulate_episode(env, optimal_policy)
    if reward == 1:
        n_win_optimal += 1
    elif reward == -1:
        n_lose_optimal += 1
    reward = simulate_episode(env, hold_policy)
    if reward == 1:
        n_win_simple += 1
    elif reward == -1:
        n_lose_simple += 1



print('Winning probability under the simple policy: {}'.format(n_win_simple/n_episode))
print('Winning probability under the optimal policy: {}'.format(n_win_optimal/n_episode))

print('Losing probability under the simple policy: {}'.format(n_lose_simple/n_episode))
print('Losing probability under the optimal policy: {}'.format(n_lose_optimal/n_episode))
