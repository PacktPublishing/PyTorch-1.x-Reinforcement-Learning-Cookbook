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

gamma = .5

threshold = 0.0001

policy_optimal = torch.tensor([[1.0, 0.0],
                               [1.0, 0.0],
                               [1.0, 0.0]])


def policy_evaluation(policy, trans_matrix, rewards, gamma, threshold):
    """
    Perform policy evaluation
    @param policy: policy matrix containing actions and their probability in each state
    @param trans_matrix: transformation matrix
    @param rewards: rewards for each state
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: values of the given policy for all possible states
    """
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state, actions in enumerate(policy):
            for action, action_prob in enumerate(actions):
                V_temp[state] += action_prob * (R[state] + gamma * torch.dot(trans_matrix[state, action], V))
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if max_delta <= threshold:
            break
    return V


V = policy_evaluation(policy_optimal, T, R, gamma, threshold)
print("The value function under the optimal policy is:\n{}".format(V))


policy_random = torch.tensor([[0.5, 0.5],
                              [0.5, 0.5],
                              [0.5, 0.5]])

V = policy_evaluation(policy_random, T, R, gamma, threshold)
print("The value function under the random policy is:\n{}".format(V))





def policy_evaluation_history(policy, trans_matrix, rewards, gamma, threshold):
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    V_his = [V]
    i = 0
    while True:
        V_temp = torch.zeros(n_state)
        i += 1
        for state, actions in enumerate(policy):
            for action, action_prob in enumerate(actions):
                V_temp[state] += action_prob * (R[state] + gamma * torch.dot(trans_matrix[state, action], V))
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        V_his.append(V)
        if max_delta <= threshold:
            break
    return V, V_his


V, V_history = policy_evaluation_history(policy_optimal, T, R, gamma, threshold)

import matplotlib.pyplot as plt
s0, = plt.plot([v[0] for v in V_history])
s1, = plt.plot([v[1] for v in V_history])
s2, = plt.plot([v[2] for v in V_history])
plt.title('Optimal policy with gamma = {}'.format(str(gamma)))
plt.xlabel('Iteration')
plt.ylabel('Policy values')
plt.legend([s0, s1, s2],
           ["State s0",
            "State s1",
            "State s2"], loc="upper left")
plt.show()
