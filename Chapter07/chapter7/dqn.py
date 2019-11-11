'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 7: Deep Q-Networks in Action
Author: Yuxi (Hayden) Liu
'''

import gym
import torch

from torch.autograd import Variable
import random


env = gym.envs.make("MountainCar-v0")



class DQN():
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(n_state, n_hidden),
                        torch.nn.ReLU(),
                        torch.nn.Linear(n_hidden, n_action)
                )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)


    def update(self, s, y):
        """
        Update the weights of the DQN given a training sample
        @param s: state
        @param y: target value
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



def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()
    return policy_function


def q_learning(env, estimator, n_episode, gamma=1.0, epsilon=0.1, epsilon_decay=.99):
    """
    Deep Q-Learning using DQN
    @param env: Gym environment
    @param estimator: DQN object
    @param n_episode: number of episodes
    @param gamma: the discount factor
    @param epsilon: parameter for epsilon_greedy
    @param epsilon_decay: epsilon decreasing factor
    """
    for episode in range(n_episode):
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        state = env.reset()
        is_done = False

        while not is_done:
            action = policy(state)
            next_state, reward, is_done, _ = env.step(action)
            total_reward_episode[episode] += reward

            modified_reward = next_state[0] + 0.5

            if next_state[0] >= 0.5:
                modified_reward += 100
            elif next_state[0] >= 0.25:
                modified_reward += 20
            elif next_state[0] >= 0.1:
                modified_reward += 10
            elif next_state[0] >= 0:
                modified_reward += 5

            q_values = estimator.predict(state).tolist()

            if is_done:
                q_values[action] = modified_reward
                estimator.update(state, q_values)
                break

            q_values_next = estimator.predict(next_state)

            q_values[action] = modified_reward + gamma * torch.max(q_values_next).item()

            estimator.update(state, q_values)

            state = next_state


        print('Episode: {}, total reward: {}, epsilon: {}'.format(episode, total_reward_episode[episode], epsilon))

        epsilon = max(epsilon * epsilon_decay, 0.01)

n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_hidden = 50
lr = 0.001
dqn = DQN(n_state, n_action, n_hidden, lr)


n_episode = 1000

total_reward_episode = [0] * n_episode

q_learning(env, dqn, n_episode, gamma=.9, epsilon=.3)



import matplotlib.pyplot as plt
plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()


