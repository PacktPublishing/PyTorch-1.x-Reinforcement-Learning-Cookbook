'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 1: Getting started with reinforcement learning and PyTorch
Author: Yuxi (Hayden) Liu
'''

import gym


env = gym.make('SpaceInvaders-v0')

env.reset()

env.render()

action = env.action_space.sample()
new_state, reward, is_done, info = env.step(action)
print(is_done)
print(info)
env.render()


is_done = False
while not is_done:
    action = env.action_space.sample()
    new_state, reward, is_done, info = env.step(action)
    print(info)
    env.render()

print(info)

print(env.action_space)

print(new_state.shape)



