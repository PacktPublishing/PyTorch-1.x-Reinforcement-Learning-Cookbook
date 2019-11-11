'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 9: Capstone Project: Playing Flappy Bird with DQN
Author: Yuxi (Hayden) Liu
'''

import random
import torch
from collections import deque

from dqn import DQN
from flappy_bird import *
from utils import pre_processing




def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()
    return policy_function


image_size = 84
batch_size = 32
lr = 1e-6
gamma = 0.99
init_epsilon = 0.1
final_epsilon = 1e-4
n_iter = 2000000
memory_size = 50000
saved_path = 'trained_models'
n_action = 2


torch.manual_seed(123)

estimator = DQN(n_action)

memory = deque(maxlen=memory_size)

env = FlappyBird()
image, reward, is_done = env.next_step(0)
image = pre_processing(image[:screen_width, :int(env.base_y)], image_size, image_size)
image = torch.from_numpy(image)
state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

for iter in range(n_iter):
    epsilon = final_epsilon + (n_iter - iter) * (init_epsilon - final_epsilon) / n_iter
    policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
    action = policy(state)
    next_image, reward, is_done = env.next_step(action)
    next_image = pre_processing(next_image[:screen_width, :int(env.base_y)], image_size, image_size)
    next_image = torch.from_numpy(next_image)
    next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
    memory.append([state, action, next_state, reward, is_done])
    loss = estimator.replay(memory, batch_size, gamma)
    state = next_state
    print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}".format(
            iter + 1, n_iter, action, loss, epsilon, reward))
    if iter+1 % 10000 == 0:
        torch.save(estimator.model, "{}/{}".format(saved_path, iter+1))

torch.save(estimator.model, "{}/final".format(saved_path))


