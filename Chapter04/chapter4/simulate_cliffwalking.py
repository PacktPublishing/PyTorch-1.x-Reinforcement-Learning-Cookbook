'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 4: Temporal Difference and Q-Learning
Author: Yuxi (Hayden) Liu
'''

import gym


env = gym.make("CliffWalking-v0")



n_state = env.observation_space.n
print(n_state)
n_action = env.action_space.n
print(n_action)


env.reset()

env.render()


new_state, reward, is_done, info = env.step(2)
env.render()

print(new_state)
print(reward)
print(is_done)
print(info)



new_state, reward, is_done, info = env.step(0)
env.render()


print(new_state)
print(reward)




new_state, reward, is_done, info = env.step(1)
new_state, reward, is_done, info = env.step(2)
env.render()
print(new_state)
print(reward)
print(is_done)



new_state, reward, is_done, info = env.step(0)
for _ in range(11):
    env.step(1)
new_state, reward, is_done, info = env.step(2)
env.render()


print(new_state)
print(reward)
print(is_done)
