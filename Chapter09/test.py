'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 9: Capstone Project: Playing Flappy Bird with DQN
Author: Yuxi (Hayden) Liu
'''

import torch
from flappy_bird import *
from utils import pre_processing


saved_path = 'trained_models'
model = torch.load("{}/final".format(saved_path))


image_size = 84
n_episode = 100



for episode in range(n_episode):
    env = FlappyBird()
    image, reward, is_done = env.next_step(0)
    image = pre_processing(image[:screen_width, :int(env.base_y)], image_size, image_size)
    image = torch.from_numpy(image)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    while True:
        prediction = model(state)[0]
        action = torch.argmax(prediction).item()

        next_image, reward, is_done = env.next_step(action)

        if is_done:
            break

        next_image = pre_processing(next_image[:screen_width, :int(env.base_y)], image_size, image_size)
        next_image = torch.from_numpy(next_image)
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        state = next_state




