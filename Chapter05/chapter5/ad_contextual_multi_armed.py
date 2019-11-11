'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 5: Solving Multi-armed Bandit Problems
Author: Yuxi (Hayden) Liu
'''

import torch
from multi_armed_bandit import BanditEnv


bandit_payout_machines = [
    [0.01, 0.015, 0.03],
    [0.025, 0.01, 0.015]
]
bandit_reward_machines = [
    [1, 1, 1],
    [1, 1, 1]
]
n_machine = len(bandit_payout_machines)

bandit_env_machines = [BanditEnv(bandit_payout, bandit_reward)
                       for bandit_payout, bandit_reward in
                       zip(bandit_payout_machines, bandit_reward_machines)]

n_episode = 100000
n_action = len(bandit_payout_machines[0])
action_count = torch.zeros(n_machine, n_action)
action_total_reward = torch.zeros(n_machine, n_action)
action_avg_reward = [[[] for action in range(n_action)] for _ in range(n_machine)]



def upper_confidence_bound(Q, action_count, t):
    ucb = torch.sqrt((2 * torch.log(torch.tensor(float(t)))) / action_count) + Q
    return torch.argmax(ucb)



Q_machines = torch.empty(n_machine, n_action)

for episode in range(n_episode):
    state = torch.randint(0, n_machine, (1,)).item()

    action = upper_confidence_bound(Q_machines[state], action_count[state], episode)
    reward = bandit_env_machines[state].step(action)
    action_count[state][action] += 1
    action_total_reward[state][action] += reward
    Q_machines[state][action] = action_total_reward[state][action] / action_count[state][action]

    for a in range(n_action):
        if action_count[state][a]:
            action_avg_reward[state][a].append(action_total_reward[state][a] / action_count[state][a])
        else:
            action_avg_reward[state][a].append(0)


import matplotlib.pyplot as plt
for state in range(n_machine):
    for action in range(n_action):
        plt.plot(action_avg_reward[state][action])
    plt.legend(['Arm {}'.format(action) for action in range(n_action)])
    plt.xscale('log')
    plt.title('Average reward over time for state {}'.format(state))
    plt.xlabel('Episode')
    plt.ylabel('Average reward')
    plt.show()

