'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 3: Monte Carlo Methods For Making Numerical Estimations
Author: Yuxi (Hayden) Liu
'''

import torch
import math
import matplotlib.pyplot as plt

n_point = 1000
points = torch.rand((n_point, 2)) * 2 - 1

n_point_circle = 0
points_circle = []

for point in points:
    r = torch.sqrt(point[0] ** 2 + point[1] ** 2)
    if r <= 1:
        points_circle.append(point)
        n_point_circle += 1

points_circle = torch.stack(points_circle)

plt.plot(points[:, 0].numpy(), points[:, 1].numpy(), 'y.')
plt.plot(points_circle[:, 0].numpy(), points_circle[:, 1].numpy(), 'c.')

i = torch.linspace(0, 2 * math.pi)
plt.plot(torch.cos(i).numpy(), torch.sin(i).numpy())
plt.axes().set_aspect('equal')
plt.show()

pi_estimated = 4 * (n_point_circle / n_point)
print('Estimated value of pi is:', pi_estimated)



def estimate_pi_mc(n_iteration):
    n_point_circle = 0
    pi_iteration = []
    for i in range(1, n_iteration+1):
        point = torch.rand(2) * 2 - 1
        r = torch.sqrt(point[0] ** 2 + point[1] ** 2)
        if r <= 1:
            n_point_circle += 1
        pi_iteration.append(4 * (n_point_circle / i))
    plt.plot(pi_iteration)
    plt.plot([math.pi] * n_iteration, '--')
    plt.xlabel('Iteration')
    plt.ylabel('Estimated pi')
    plt.title('Estimation history')
    plt.show()
    print('Estimated value of pi is:', pi_iteration[-1])

estimate_pi_mc(10000)