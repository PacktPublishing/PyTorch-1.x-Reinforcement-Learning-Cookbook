'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 8: Implementing Policy Gradients and Policy Optimization
Author: Yuxi (Hayden) Liu
'''

import torch
import gym
import torch.nn as nn
from torch.autograd import Variable


env = gym.make('CartPole-v0')

class PolicyNetwork():
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.001):

        self.model = nn.Sequential(
                        nn.Linear(n_state, n_hidden),
                        nn.ReLU(),
                        nn.Linear(n_hidden, n_action),
                        nn.Softmax(),
                )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)


    def predict(self, s):
        """
        Compute the action probabilities of state s using the learning model
        @param s: input state
        @return: predicted policy
        """
        return self.model(torch.Tensor(s))


    def update(self, advantages, log_probs):
        """
        Update the weights of the policy network given the training samples
        @param advantages: advantage for each step in an episode
        @param log_probs: log probability for each step
        """
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, advantages):
            policy_gradient.append(-log_prob * Gt)

        loss = torch.stack(policy_gradient).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_action(self, s):
        """
        Estimate the policy and sample an action, compute its log probability
        @param s: input state
        @return: the selected action and log probability
        """
        probs = self.predict(s)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action])
        return action, log_prob




class ValueNetwork():
    def __init__(self, n_state, n_hidden=50, lr=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(n_state, n_hidden),
                        torch.nn.ReLU(),
                        torch.nn.Linear(n_hidden, 1)
                )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)


    def update(self, s, y):
        """
        Update the weights of the DQN given a training sample
        @param s: states
        @param y: target values
        """
        y_pred = self.model(torch.Tensor(s))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self, s):
        """
        Compute the Q values of the state for all actions using the learning model
        @param s: input state
        @return: Q values of the state for all actions
        """
        with torch.no_grad():
            return self.model(torch.Tensor(s))


def reinforce(env, estimator_policy, estimator_value, n_episode, gamma=1.0):
    """
    REINFORCE algorithm with baseline
    @param env: Gym environment
    @param estimator_policy: policy network
    @param estimator_value: value network
    @param n_episode: number of episodes
    @param gamma: the discount factor
    """
    for episode in range(n_episode):
        log_probs = []
        states = []
        rewards = []
        state = env.reset()

        while True:
            states.append(state)
            action, log_prob = estimator_policy.get_action(state)
            next_state, reward, is_done, _ = env.step(action)

            total_reward_episode[episode] += reward
            log_probs.append(log_prob)

            rewards.append(reward)

            if is_done:
                Gt = 0
                pw = 0

                returns = []
                for t in range(len(states)-1, -1, -1):
                    Gt += gamma ** pw * rewards[t]
                    pw += 1
                    returns.append(Gt)


                returns = returns[::-1]
                returns = torch.tensor(returns)

                baseline_values = estimator_value.predict(states)

                advantages = returns - baseline_values


                estimator_value.update(states, returns)

                estimator_policy.update(advantages, log_probs)


                print('Episode: {}, total reward: {}'.format(episode, total_reward_episode[episode]))
                break


            state = next_state


n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_hidden_p = 64
lr_p = 0.003
policy_net = PolicyNetwork(n_state, n_action, n_hidden_p, lr_p)

n_hidden_v = 64
lr_v = 0.003
value_net = ValueNetwork(n_state, n_hidden_v, lr_v)

n_episode = 2000
gamma = 0.9
total_reward_episode = [0] * n_episode

reinforce(env, policy_net, value_net, n_episode, gamma)

import matplotlib.pyplot as plt
plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
