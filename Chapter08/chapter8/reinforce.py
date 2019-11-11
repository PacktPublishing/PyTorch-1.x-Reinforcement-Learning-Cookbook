'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 8: Implementing Policy Gradients and Policy Optimization
Author: Yuxi (Hayden) Liu
'''

import torch
import gym
import torch.nn as nn


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


    def update(self, returns, log_probs):
        """
        Update the weights of the policy network given the training samples
        @param returns: return (cumulative rewards) for each step in an episode
        @param log_probs: log probability for each step
        """
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, returns):
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




def reinforce(env, estimator, n_episode, gamma=1.0):
    """
    REINFORCE algorithm
    @param env: Gym environment
    @param estimator: policy network
    @param n_episode: number of episodes
    @param gamma: the discount factor
    """
    for episode in range(n_episode):
        log_probs = []
        rewards = []
        state = env.reset()

        while True:
            action, log_prob = estimator.get_action(state)
            next_state, reward, is_done, _ = env.step(action)

            total_reward_episode[episode] += reward
            log_probs.append(log_prob)
            rewards.append(reward)

            if is_done:

                returns = []

                Gt = 0
                pw = 0

                for reward in rewards[::-1]:
                    Gt += gamma ** pw * reward
                    pw += 1
                    returns.append(Gt)

                returns = returns[::-1]
                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) / (
                    returns.std() + 1e-9)

                estimator.update(returns, log_probs)
                print('Episode: {}, total reward: {}'.format(episode, total_reward_episode[episode]))
                break

            state = next_state



n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_hidden = 128
lr = 0.003
policy_net = PolicyNetwork(n_state, n_action, n_hidden, lr)


n_episode = 500
gamma = 0.9
total_reward_episode = [0] * n_episode

reinforce(env, policy_net, n_episode, gamma)

import matplotlib.pyplot as plt
plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
