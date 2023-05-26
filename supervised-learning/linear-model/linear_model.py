import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod


class AbstractRegressor(ABC):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def load_dataset(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)

    @abstractmethod
    def train_model(self):
        pass

    def evaluate_model(self):
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)
        print("Training set score: ", train_score)
        print("Test set score: ", test_score)


class OLSRegressor(AbstractRegressor):
    def train_model(self):
        self.model = linear_model.LinearRegression()
        self.model.fit(self.X_train, self.y_train)


class RidgeRegressor(AbstractRegressor):
    def __init__(self, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def train_model(self):
        self.model = linear_model.Ridge(alpha=self.alpha)
        self.model.fit(self.X_train, self.y_train)


class LassoRegressor(AbstractRegressor):
    def __init__(self, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def train_model(self):
        self.model = linear_model.Lasso(alpha=self.alpha)
        self.model.fit(self.X_train, self.y_train)
