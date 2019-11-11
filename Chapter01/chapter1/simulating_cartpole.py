'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 1: Getting started with reinforcement learning and PyTorch
Author: Yuxi (Hayden) Liu
'''


import gym
env = gym.make('CartPole-v0')

video_dir = './cartpole_video/'
env = gym.wrappers.Monitor(env, video_dir)

env.reset()

env.render()


is_done = False
while not is_done:
    action = env.action_space.sample()
    new_state, reward, is_done, info = env.step(action)
    print(new_state)
    env.render()



n_episode = 10000
total_rewards = []
for episode in range(n_episode):
    state = env.reset()
    total_reward = 0
    is_done = False
    while not is_done:
        action = env.action_space.sample()
        state, reward, is_done, _ = env.step(action)
        total_reward += reward
    total_rewards.append(total_reward)


print('Average total reward over {} episodes: {}'.format(n_episode, sum(total_rewards) / n_episode))