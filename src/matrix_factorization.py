import numpy as np
import pandas as pd
import os


class SVD:
    def __init__(self, lrate=0.01, k=5, epoch=10, reg=0.02):
        self.lrate = lrate
        self.k = k
        self.epoch = epoch
        self.reg = reg
        self.u = None
        self.v = None
        self.mean_movie_ratings = None
        self.mean_user_ratings = None
        self.offset_user_ratings = None
        self.global_movie_rating = None
        self.losses = []

    def train(self, X: np.ndarray):
        self.mean_movie_ratings = np.nanmean(X, axis=0)
        self.mean_user_ratings = np.nanmean(X, axis=1)
        self.offset_user_ratings = np.nanmean(X - self.mean_movie_ratings, axis=1)
        self.global_movie_rating = np.mean(self.mean_movie_ratings)

        self.num_users, self.num_items = X.shape

        self.u = np.random.normal(scale=0.1, size=(self.num_users, self.k))
        self.v = np.random.normal(scale=0.1, size=(self.num_items, self.k))

        for epoch in range(self.epoch):
            loss = 0
            count = 0

            for i in range(self.num_users):
                for j in range(self.num_items):

                    rating = X[i, j]

                    if not np.isnan(rating):
                        pred = self.u[i, :] @ self.v[j, :].T
                        error = rating - pred

                        loss += error**2
                        count += 1

                        self.u[i, :] += self.lrate * (
                            error * self.v[j, :] - self.reg * self.u[i, :]
                        )
                        self.v[j, :] += self.lrate * (
                            error * self.u[i, :] - self.reg * self.v[j, :]
                        )

            rmse = np.sqrt(loss / count) if count > 0 else 0
            self.losses.append(rmse)

            print(f"Epoch {epoch+1}/{self.epoch} => RMSE: {rmse:.4f}")

        return self

    def predict(self, uid: np.ndarray, iid: np.ndarray):
        user_ids = uid - 1
        item_ids = iid - 1

        valid_user = user_ids < self.num_users
        valid_item = item_ids < self.num_items

        both_valid = valid_user & valid_item
        only_user = valid_user & ~valid_item
        only_item = valid_item & ~valid_user

        predicted_ratings = np.full(uid.shape[0], self.global_movie_rating)
        predicted_ratings[both_valid] = np.einsum(
            "ij,ij->i", self.u[user_ids[both_valid]], self.v[item_ids[both_valid]]
        )
        predicted_ratings[only_user] = (
            self.global_movie_rating + self.offset_user_ratings[user_ids[only_user]]
        )
        predicted_ratings[only_item] = self.mean_movie_ratings[item_ids[only_item]]

        return np.clip(predicted_ratings, a_min=0, a_max=5)
