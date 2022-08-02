import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston


class CustomLinearRegression:

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = np.array([])
        self.intercept = 0.0
        self.r2 = ...
        self.rmse_ = ...
        self.features = np.matrix

    def fit(self, X, y):
        if self.fit_intercept:
            X['ones'] = [1 for _ in range(X.shape[0])]
            self.features = np.matrix(X.values)
            X = self.features
            y = np.matrix(y.values)
            beta = np.linalg.inv(X.T @ X) @ X.T @ y
            n = beta.shape[0]
            self.intercept = beta[n-1, 0]
            self.coefficient = np.array(beta[:n-1])
        else:
            self.features = np.matrix(X.values)
            X = self.features
            y = np.matrix(y.values)
            self.coefficient = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        return np.matrix(X.values) @ np.matrix(self.coefficient) + self.intercept

    def r2_score(self, y, yhat):
        self.r2 = (1 - (np.sum(np.subtract(y, yhat) ** 2) / np.sum(np.subtract(y, y.mean()) ** 2))).array[0]

    def rmse(self, y, yhat):
        self.rmse_ = ((np.sum(np.subtract(y, yhat) ** 2) / len(y)) ** 0.5).array[0]


def main():
    data = load_boston()
    X, y = data.data, data.target
    X_train = X[:-100, :]
    y_train = y[:-100]
    X_test = X[-100:, :]
    y_test = y[-100:]

    customModel = CustomLinearRegression(fit_intercept=True)
    customModel.fit(pd.DataFrame(X_train), pd.DataFrame(y_train))
    y_pred1 = customModel.predict(pd.DataFrame(X_test))
    customModel.r2_score(pd.DataFrame(y_test), y_pred1)
    customModel.rmse(pd.DataFrame(y_test), y_pred1)

    skModel = LinearRegression(fit_intercept=True)
    skModel.fit(X_train, y_train)
    y_pred2 = skModel.predict(X_test)
    r2_sk = r2_score(y_test, y_pred2)
    rmse_sk = mean_squared_error(y_test, y_pred2) ** 0.5

    output = {'Intercept': skModel.intercept_ - customModel.intercept,
              'Coefficient': skModel.coef_ - customModel.coefficient.T,
              'R2': r2_sk - customModel.r2,
              'RMSE': rmse_sk - customModel.rmse_}
    print(output)


if __name__ == '__main__':
    main()
