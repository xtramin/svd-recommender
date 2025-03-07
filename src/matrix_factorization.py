import numpy as np
import pandas as pd
import os


class SVD:
    def __init__(self, lrate=0.01, k=5, epoch=1):
        self.lrate = lrate
        self.k = k
        self.epoch = epoch
        self.u = None
        self.v = None

    def train(self, X):
        num_users, num_items = X.shape

        self.u = np.random.rand(num_users, self.k)
        self.v = np.random.rand(num_items, self.k)

        for _ in range(self.epoch):
            for i in range(num_users):
                for j in range(num_items):

                    rating = X[i, j]

                    if not np.isnan(rating):
                        error = rating - self.u[i, :] @ self.v[j, :].T

                        for f in range(self.k):
                            self.u[i, f] += self.lrate * error * self.v[j, f]
                            self.v[j, f] += self.lrate * error * self.u[i, f]

        return self.u, self.v