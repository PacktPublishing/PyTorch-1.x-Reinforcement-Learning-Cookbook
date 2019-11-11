'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 6: Scaling up Learning with Function Approximation
Author: Yuxi (Hayden) Liu
'''


import gym

env = gym.envs.make("MountainCar-v0")



n_action = env.action_space.n
print(n_action)


env.reset()

is_done = False
while not is_done:
    next_state, reward, is_done, info = env.step(2)
    print(next_state, reward, is_done)
    env.render()

env.close()