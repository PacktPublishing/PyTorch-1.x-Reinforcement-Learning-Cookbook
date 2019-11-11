'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 6: Scaling up Learning with Function Approximation
Author: Yuxi (Hayden) Liu
'''

import torch
from torch.autograd import Variable
import math


class Estimator():
    def __init__(self, n_feat, n_state, n_action, lr=0.05):
        self.w, self.b = self.get_gaussian_wb(n_feat, n_state)
        self.n_feat = n_feat
        self.models = []
        self.optimizers = []
        self.criterion = torch.nn.MSELoss()

        for _ in range(n_action):
            model = torch.nn.Linear(n_feat, 1)
            self.models.append(model)
            optimizer = torch.optim.SGD(model.parameters(), lr)
            self.optimizers.append(optimizer)


    def get_gaussian_wb(self, n_feat, n_state, sigma=.2):
        """
        Generate the coefficients of the feature set from Gaussian distribution
        @param n_feat: number of features
        @param n_state: number of states
        @param sigma: kernel parameter
        @return: coefficients of the features
        """
        torch.manual_seed(0)
        w = torch.randn((n_state, n_feat)) * 1.0 / sigma
        b = torch.rand(n_feat) * 2.0 * math.pi
        return w, b

    def get_feature(self, s):
        """
        Generate features based on the input state
        @param s: input state
        @return: features
        """
        features = (2.0 / self.n_feat) ** .5 * torch.cos(
            torch.matmul(torch.tensor(s).float(), self.w) + self.b)
        return features


    def update(self, s, a, y):
        """
        Update the weights for the linear estimator with the given training sample
        @param s: state
        @param a: action
        @param y: target value
        """
        features = Variable(self.get_feature(s))
        y_pred = self.models[a](features)

        loss = self.criterion(y_pred, Variable(torch.Tensor([y])))

        self.optimizers[a].zero_grad()
        loss.backward()
        self.optimizers[a].step()

    def predict(self, s):
        """
        Compute the Q values of the state using the learning model
        @param s: input state
        @return: Q values of the state
        """
        features = self.get_feature(s)
        with torch.no_grad():
            return torch.tensor([model(features) for model in self.models])


if __name__ == "__main__":

    estimator = Estimator(10, 2, 1)
    s1 = [0.5, 0.1]
    print(estimator.get_feature(s1))


    s_list = [[1, 2], [2, 2], [3, 4], [2, 3], [2, 1]]
    target_list = [1, 1.5, 2, 2, 1.5]

    for s, target in zip(s_list, target_list):
        feature = estimator.get_feature(s)
        estimator.update(s, 0, target)

    print(estimator.predict([0.5, 0.1]))
    print(estimator.predict([2, 3]))
