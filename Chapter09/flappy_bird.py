'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 9: Capstone Project: Playing Flappy Bird with DQN
Author: Yuxi (Hayden) Liu
'''

from itertools import cycle
from random import randint
import pygame

from utils import load_images


pygame.init()
fps = 30
fps_clock = pygame.time.Clock()

screen_width = 288
screen_height = 512
screen = pygame.display.set_mode((screen_width, screen_height))

pygame.display.set_caption('Flappy Bird')

base_image, background_image, pipe_images, bird_images, bird_hitmask, pipe_hitmask = load_images('sprites/')

bird_width = bird_images[0].get_width()
bird_height = bird_images[0].get_height()
pipe_width = pipe_images[0].get_width()
pipe_height = pipe_images[0].get_height()



pipe_gap_size = 100
bird_index_gen = cycle([0, 1, 2, 1])



class FlappyBird(object):
    def __init__(self):
        self.pipe_vel_x = -4
        self.min_velocity_y = -8
        self.max_velocity_y = 10
        self.downward_speed = 1
        self.upward_speed = -9
        self.cur_velocity_y = 0
        self.iter = self.bird_index = self.score = 0
        self.bird_x = int(screen_width / 5)
        self.bird_y = int((screen_height - bird_height) / 2)
        self.base_x = 0
        self.base_y = screen_height * 0.79
        self.base_shift = base_image.get_width() - background_image.get_width()
        self.pipes = [self.gen_random_pipe(screen_width), self.gen_random_pipe(screen_width * 1.5)]
        self.is_flapped = False


    def gen_random_pipe(self, x):
        gap_y = randint(2, 10) * 10 + int(self.base_y * 0.2)
        return {"x_upper": x,
                "y_upper": gap_y - pipe_height,
                "x_lower": x,
                "y_lower": gap_y + pipe_gap_size}

    def check_collision(self):
        if bird_height + self.bird_y >= self.base_y - 1:
            return True
        bird_rect = pygame.Rect(self.bird_x, self.bird_y, bird_width, bird_height)
        for pipe in self.pipes:
            pipe_boxes = [pygame.Rect(pipe["x_upper"], pipe["y_upper"], pipe_width, pipe_height),
                          pygame.Rect(pipe["x_lower"], pipe["y_lower"], pipe_width, pipe_height)]
            # Check if the bird's bounding box overlaps to the bounding box of any pipe
            if bird_rect.collidelist(pipe_boxes) == -1:
                return False
            for i in range(2):
                cropped_bbox = bird_rect.clip(pipe_boxes[i])
                x1 = cropped_bbox.x - bird_rect.x
                y1 = cropped_bbox.y - bird_rect.y
                x2 = cropped_bbox.x - pipe_boxes[i].x
                y2 = cropped_bbox.y - pipe_boxes[i].y
                for x in range(cropped_bbox.width):
                    for y in range(cropped_bbox.height):
                        if bird_hitmask[self.bird_index][x1+x, y1+y] and pipe_hitmask[i][x2+x, y2+y]:
                            return True
        return False

    def next_step(self, action):
        pygame.event.pump()
        reward = 0.1
        if action == 1:
            self.cur_velocity_y = self.upward_speed
            self.is_flapped = True
        # Update score
        bird_center_x = self.bird_x + bird_width / 2
        for pipe in self.pipes:
            pipe_center_x = pipe["x_upper"] + pipe_width / 2
            if pipe_center_x < bird_center_x < pipe_center_x + 5:
                self.score += 1
                reward = 1
                break
        # Update index and iteration
        if (self.iter + 1) % 3 == 0:
            self.bird_index = next(bird_index_gen)
        self.iter = (self.iter + 1) % fps
        self.base_x = -((-self.base_x + 100) % self.base_shift)
        # Update bird's position
        if self.cur_velocity_y < self.max_velocity_y and not self.is_flapped:
            self.cur_velocity_y += self.downward_speed
        self.is_flapped = False
        self.bird_y += min(self.cur_velocity_y, self.bird_y - self.cur_velocity_y - bird_height)
        if self.bird_y < 0:
            self.bird_y = 0
        # Update pipe position
        for pipe in self.pipes:
            pipe["x_upper"] += self.pipe_vel_x
            pipe["x_lower"] += self.pipe_vel_x
        #  Add new pipe when first pipe is about to touch left of screen
        if 0 < self.pipes[0]["x_lower"] < 5:
            self.pipes.append(self.gen_random_pipe(screen_width + 10))
        # remove first pipe if its out of the screen
        if self.pipes[0]["x_lower"] < -pipe_width:
            self.pipes.pop(0)
        if self.check_collision():
            is_done = True
            reward = -1
            self.__init__()
        else:
            is_done = False
        # Draw sprites
        screen.blit(background_image, (0, 0))
        screen.blit(base_image, (self.base_x, self.base_y))
        screen.blit(bird_images[self.bird_index], (self.bird_x, self.bird_y))
        for pipe in self.pipes:
            screen.blit(pipe_images[0], (pipe["x_upper"], pipe["y_upper"]))
            screen.blit(pipe_images[1], (pipe["x_lower"], pipe["y_lower"]))
        image = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        fps_clock.tick(fps)
        return image, reward, is_done

