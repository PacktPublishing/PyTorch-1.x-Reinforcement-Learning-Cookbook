'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 7: Deep Q-Networks in Action
Author: Yuxi (Hayden) Liu
'''

import gym
import torch
import random

env = gym.envs.make("PongDeterministic-v4")

state_shape = env.observation_space.shape
n_action = env.action_space.n

print(state_shape)
print(n_action)
print(env.unwrapped.get_action_meanings())

ACTIONS = [0, 2, 3]
n_action = 3
env.reset()

is_done = False
while not is_done:
    action = ACTIONS[random.randint(0, n_action - 1)]
    obs, reward, is_done, _ = env.step(action)
    env.render()

import torchvision.transforms as T
from PIL import Image

image_size = 84


transform = T.Compose([T.ToPILImage(),
                       T.Grayscale(num_output_channels=1),
                       T.Resize((image_size, image_size), interpolation=Image.CUBIC),
                       T.ToTensor(),
                       ])


def get_state(obs):
    state = obs.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    state = transform(state)
    return state


state = get_state(obs)


print(state.shape)


from collections import deque
import copy
from torch.autograd import Variable


class DQN():
    def __init__(self, n_state, n_action, n_hidden, lr=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_hidden[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden[0], n_hidden[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden[1], n_action)
        )
        self.model_target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, s, y):
        """
        Update the weights of the DQN given a training sample
        @param s: state
        @param y: target value
        """
        y_pred = self.model(torch.Tensor(s))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        """
        Compute the Q values of the state for all actions using the learning model
        @param s: input state
        @return: Q values of the state for all actions
        """
        with torch.no_grad():
            return self.model(torch.Tensor(s))

    def target_predict(self, s):
        """
        Compute the Q values of the state for all actions using the target network
        @param s: input state
        @return: targeted Q values of the state for all actions
        """
        with torch.no_grad():
            return self.model_target(torch.Tensor(s))

    def replay(self, memory, replay_size, gamma):
        """
        Experience replay with target network
        @param memory: a list of experience
        @param replay_size: the number of samples we use to update the model each time
        @param gamma: the discount factor
        """
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            states = []
            td_targets = []
            for state, action, next_state, reward, is_done in replay_data:
                states.append(state.tolist())
                q_values = self.predict(state).tolist()
                if is_done:
                    q_values[action] = reward
                else:
                    q_values_next = self.target_predict(next_state).detach()

                    q_values[action] = reward + gamma * torch.max(q_values_next).item()

                td_targets.append(q_values)

            self.update(states, td_targets)

    def copy_target(self):
        self.model_target.load_state_dict(self.model.state_dict())


def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()

    return policy_function


def q_learning(env, estimator, n_episode, replay_size, target_update=10, gamma=1.0, epsilon=0.1, epsilon_decay=.99):
    """
    Deep Q-Learning using double DQN, with experience replay
    @param env: Gym environment
    @param estimator: DQN object
    @param replay_size: number of samples we use to update the model each time
    @param target_update: number of episodes before updating the target network
    @param n_episode: number of episodes
    @param gamma: the discount factor
    @param epsilon: parameter for epsilon_greedy
    @param epsilon_decay: epsilon decreasing factor
    """
    for episode in range(n_episode):
        if episode % target_update == 0:
            estimator.copy_target()
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        obs = env.reset()

        state = get_state(obs).view(image_size * image_size)

        is_done = False
        while not is_done:
            action = policy(state)
            next_obs, reward, is_done, _ = env.step(ACTIONS[action])

            total_reward_episode[episode] += reward

            next_state = get_state(obs).view(image_size * image_size)

            memory.append((state, action, next_state, reward, is_done))

            if is_done:
                break

            estimator.replay(memory, replay_size, gamma)

            state = next_state

        print('Episode: {}, total reward: {}, epsilon: {}'.format(episode, total_reward_episode[episode], epsilon))
        epsilon = max(epsilon * epsilon_decay, 0.01)


n_state = image_size * image_size

n_episode = 1000

n_hidden = [200, 50]
lr = 0.003
replay_size = 32
target_update = 10

dqn = DQN(n_state, n_action, n_hidden, lr)
memory = deque(maxlen=10000)
total_reward_episode = [0] * n_episode

q_learning(env, dqn, n_episode, replay_size, target_update, gamma=.9, epsilon=1)
