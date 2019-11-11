'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 5: Solving Multi-armed Bandit Problems
Author: Yuxi (Hayden) Liu
'''

import torch
import matplotlib.pyplot as plt


beta1 = torch.distributions.beta.Beta(1, 1)
samples1 = [beta1.sample() for _ in range(100000)]
plt.hist(samples1, range=[0, 1], bins=10)
plt.title('beta(1, 1)')
plt.show()

beta2 = torch.distributions.beta.Beta(5, 1)
samples2 = [beta2.sample() for _ in range(100000)]
plt.hist(samples2, range=[0, 1], bins=10)
plt.title('beta(5, 1)')
plt.show()

beta3 = torch.distributions.beta.Beta(1, 5)
samples3= [beta3.sample() for _ in range(100000)]
plt.hist(samples3, range=[0, 1], bins=10)
plt.title('beta(1, 5)')
plt.show()

beta4 = torch.distributions.beta.Beta(5, 5)
samples4= [beta4.sample() for _ in range(100000)]
plt.hist(samples4, range=[0, 1], bins=10)
plt.title('beta(5, 5)')
plt.show()



from multi_armed_bandit import BanditEnv


bandit_payout = [0.01, 0.015, 0.03]
bandit_reward = [1, 1, 1]
bandit_env = BanditEnv(bandit_payout, bandit_reward)

n_episode = 100000
n_action = len(bandit_payout)
action_count = torch.tensor([0. for _ in range(n_action)])
action_total_reward = [0 for _ in range(n_action)]
action_avg_reward = [[] for action in range(n_action)]



def thompson_sampling(alpha, beta):
    prior_values = torch.distributions.beta.Beta(alpha, beta).sample()
    return torch.argmax(prior_values)


alpha = torch.ones(n_action)
beta = torch.ones(n_action)


for episode in range(n_episode):
    action = thompson_sampling(alpha, beta)
    reward = bandit_env.step(action)
    action_count[action] += 1
    action_total_reward[action] += reward

    if reward > 0:
        alpha[action] += 1
    else:
        beta[action] += 1

    for a in range(n_action):
        if action_count[a]:
            action_avg_reward[a].append(action_total_reward[a] / action_count[a])
        else:
            action_avg_reward[a].append(0)



for action in range(n_action):
    plt.plot(action_avg_reward[action])

plt.legend(['Arm {}'.format(action) for action in range(n_action)])
plt.xscale('log')
plt.title('Average reward over time')
plt.xlabel('Episode')
plt.ylabel('Average reward')
plt.show()



