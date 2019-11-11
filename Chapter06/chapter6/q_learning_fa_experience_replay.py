'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 6: Scaling up Learning with Function Approximation
Author: Yuxi (Hayden) Liu
'''

import gym
import torch
from linear_estimator import Estimator
from collections import deque
import random

env = gym.envs.make("MountainCar-v0")



def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        probs = torch.ones(n_action) * epsilon / n_action
        q_values = estimator.predict(state)
        best_action = torch.argmax(q_values).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function


def q_learning(env, estimator, n_episode, replay_size, gamma=1.0, epsilon=0.1, epsilon_decay=.99):
    """
    Q-Learning algorithm using Function Approximation, with experience replay
    @param env: Gym environment
    @param estimator: Estimator object
    @param replay_size: number of samples we use to update the model each time
    @param n_episode: number of episodes
    @param gamma: the discount factor
    @param epsilon: parameter for epsilon_greedy
    @param epsilon_decay: epsilon decreasing factor
    """
    for episode in range(n_episode):
        policy = gen_epsilon_greedy_policy(estimator, epsilon * epsilon_decay ** episode, n_action)
        state = env.reset()
        is_done = False
        while not is_done:
            action = policy(state)
            next_state, reward, is_done, _ = env.step(action)
            total_reward_episode[episode] += reward

            if is_done:
                break

            q_values_next = estimator.predict(next_state)
            td_target = reward + gamma * torch.max(q_values_next)

            memory.append((state, action, td_target))

            state = next_state

        replay_data = random.sample(memory, replay_size)

        for state, action, td_target in replay_data:
            estimator.update(state, action, td_target)



n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_feature = 200
lr = 0.03
estimator = Estimator(n_feature, n_state, n_action, lr)


memory = deque(maxlen=400)

n_episode = 1000
replay_size = 190
total_reward_episode = [0] * n_episode

q_learning(env, estimator, n_episode, replay_size, epsilon=0.1)



import matplotlib.pyplot as plt
plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()