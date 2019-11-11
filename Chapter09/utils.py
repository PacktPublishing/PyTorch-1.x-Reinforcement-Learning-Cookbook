'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 9: Capstone Project: Playing Flappy Bird with DQN
Author: Yuxi (Hayden) Liu
'''

import cv2
import numpy as np
from pygame.image import load
from pygame.surfarray import pixels_alpha
from pygame.transform import rotate



def load_images(sprites_path):
    base_image = load(sprites_path + 'base.png').convert_alpha()
    background_image = load(sprites_path + 'background-black.png').convert()
    pipe_images = [rotate(load(sprites_path + 'pipe-green.png').convert_alpha(), 180),
                   load(sprites_path + 'pipe-green.png').convert_alpha()]
    bird_images = [load(sprites_path + 'redbird-upflap.png').convert_alpha(),
                   load(sprites_path + 'redbird-midflap.png').convert_alpha(),
                   load(sprites_path + 'redbird-downflap.png').convert_alpha()]
    bird_hitmask = [pixels_alpha(image).astype(bool) for image in bird_images]
    pipe_hitmask = [pixels_alpha(image).astype(bool) for image in pipe_images]
    return base_image, background_image, pipe_images, bird_images, bird_hitmask, pipe_hitmask


def pre_processing(image, width, height):
    image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return image[None, :, :].astype(np.float32)


