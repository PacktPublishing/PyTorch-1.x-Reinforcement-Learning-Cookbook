'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 2: Markov Decision Process and Dynamic Programming
Author: Yuxi (Hayden) Liu
'''

import torch


T = torch.tensor([[[0.8, 0.1, 0.1],
                   [0.1, 0.6, 0.3]],
                  [[0.7, 0.2, 0.1],
                   [0.1, 0.8, 0.1]],
                  [[0.6, 0.2, 0.2],
                   [0.1, 0.4, 0.5]]]
                 )

R = torch.tensor([1., 0, -1.])

gamma = 0.5

action = 0

def cal_value_matrix_inversion(gamma, trans_matrix, rewards):
    inv = torch.inverse(torch.eye(rewards.shape[0]) - gamma * trans_matrix)
    V = torch.mm(inv, rewards.reshape(-1, 1))
    return V

trans_matrix = T[:, action]
V = cal_value_matrix_inversion(gamma, trans_matrix, R)
print("The value function under the optimal policy is:\n{}".format(V))


gamma = 0
V = cal_value_matrix_inversion(gamma, trans_matrix, R)
print("The value function under the optimal policy is:\n{}".format(V))

gamma = 0.99
V = cal_value_matrix_inversion(gamma, trans_matrix, R)
print("The value function under the optimal policy is:\n{}".format(V))




