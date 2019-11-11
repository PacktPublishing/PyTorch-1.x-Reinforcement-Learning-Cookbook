'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 5: Solving Multi-armed Bandit Problems
Author: Yuxi (Hayden) Liu
'''

import torch


class BanditEnv():
    """
    Multi-armed bandit environment
    payout_list:
        A list of probabilities of the likelihood that a particular bandit will pay out
    reward_list:
        A list of rewards of the payout that bandit has
    """
    def __init__(self, payout_list, reward_list):
        self.payout_list = payout_list
        self.reward_list = reward_list

    def step(self, action):
        if torch.rand(1).item() < self.payout_list[action]:
            return self.reward_list[action]
        return 0



if __name__ == "__main__":
    bandit_payout = [0.1, 0.15, 0.3]
    bandit_reward = [4, 3, 1]
    bandit_env = BanditEnv(bandit_payout, bandit_reward)

    n_episode = 100000
    n_action = len(bandit_payout)
    action_count = [0 for _ in range(n_action)]
    action_total_reward = [0 for _ in range(n_action)]
    action_avg_reward = [[] for action in range(n_action)]

    def random_policy():
        action = torch.multinomial(torch.ones(n_action), 1).item()
        return action

    for episode in range(n_episode):
        action = random_policy()
        reward = bandit_env.step(action)
        action_count[action] += 1
        action_total_reward[action] += reward
        for a in range(n_action):
            if action_count[a]:
                action_avg_reward[a].append(action_total_reward[a] / action_count[a])
            else:
                action_avg_reward[a].append(0)


    import matplotlib.pyplot as plt
    for action in range(n_action):
        plt.plot(action_avg_reward[action])

    plt.legend(['Arm {}'.format(action) for action in range(n_action)])
    plt.xscale('log')
    plt.title('Average reward over time')
    plt.xlabel('Episode')
    plt.ylabel('Average reward')
    plt.show()

