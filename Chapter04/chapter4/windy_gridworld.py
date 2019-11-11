'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 4: Temporal Difference and Q-Learning
Author: Yuxi (Hayden) Liu
'''


import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class WindyGridworldEnv(discrete.DiscreteEnv):
    def __init__(self):
        self.shape = (7, 10)
        nS = self.shape[0] * self.shape[1]
        nA = 4

        # Wind locations
        winds = np.zeros(self.shape)
        winds[:,[3,4,5,8]] = 1
        winds[:,[6,7]] = 2

        self.goal = (3, 7)

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)

        # Calculate initial state distribution
        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3,0), self.shape)] = 1.0

        super(WindyGridworldEnv, self).__init__(nS, nA, P, isd)

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        @param current: (row, col), current position on the grid
        @param delta: Change in position for transition
        @param winds: Wind effect
        @return: (1.0, new_state, reward, is_done)
        """
        new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == self.goal
        return [(1.0, new_state, -1.0, is_done)]


    def render(self):
        outfile = sys.stdout
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            elif position == self.goal:
                output = " T "
            else:
                output = " o "
            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"
            outfile.write(output)
        outfile.write("\n")


if __name__ == "__main__":

    env = WindyGridworldEnv()
    env.reset()
    env.render()

    print(env.step(1))
    env.render()

    print(env.step(1))
    env.render()

    print(env.step(1))
    env.render()

    print(env.step(1))
    env.render()

    print(env.step(1))
    env.render()

    print(env.step(1))
    env.render()

    print(env.step(1))
    env.render()

    print(env.step(1))
    env.render()

    print(env.step(1))
    env.render()

    print(env.step(2))
    env.render()

    print(env.step(2))
    env.render()

    print(env.step(2))
    env.render()

    print(env.step(2))
    env.render()

    print(env.step(3))
    env.render()

    print(env.step(3))
    env.render()
