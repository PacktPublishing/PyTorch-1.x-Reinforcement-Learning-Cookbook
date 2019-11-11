'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 9: Capstone Project: Playing Flappy Bird with DQN
Author: Yuxi (Hayden) Liu
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class DQNModel(nn.Module):
    def __init__(self, n_action=2):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc = nn.Linear(7 * 7 * 64, 512)
        self.out = nn.Linear(512, n_action)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        output = self.out(x)
        return output


class DQN():
    def __init__(self, n_action, lr=1e-6):
        self.criterion = nn.MSELoss()
        self.model = DQNModel(n_action)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, y_predict, y_target):
        """
        Update the weights of the DQN given a training sample
        @param y_predict:
        @param y_target:
        @return:
        """
        loss = self.criterion(y_predict, y_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, s):
        """
        Compute the Q values of the state for all actions using the learning model
        @param s: input state
        @return: Q values of the state for all actions
        """
        return self.model(torch.Tensor(s))

    def replay(self, memory, replay_size, gamma):
        """
        Experience replay
        @param memory: a list of experience
        @param replay_size: the number of samples we use to update the model each time
        @param gamma: the discount factor
        @return: the loss
        """
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*replay_data)

            state_batch = torch.cat(tuple(state for state in state_batch))
            next_state_batch = torch.cat(tuple(state for state in next_state_batch))
            q_values_batch = self.predict(state_batch)
            q_values_next_batch = self.predict(next_state_batch)

            reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])

            action_batch = torch.from_numpy(
                np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))

            q_value = torch.sum(q_values_batch * action_batch, dim=1)

            td_targets = torch.cat(
                tuple(reward if terminal else reward + gamma * torch.max(prediction) for reward, terminal, prediction
                    in zip(reward_batch, done_batch, q_values_next_batch)))

            loss = self.update(q_value, td_targets)
            return loss
