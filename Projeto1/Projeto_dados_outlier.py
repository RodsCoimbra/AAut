import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sys import exit

x = np.load("Projeto1/X_train_regression1.npy")
y = np.load("Projeto1/y_train_regression1.npy")
x = np.delete(x, 2, 0)
y = np.delete(y, 2, 0)
reg = linear_model.LinearRegression()
reg.fit(x, y)
x = np.load("Projeto1/X_train_regression1.npy")
y = np.load("Projeto1/y_train_regression1.npy")
y_est = reg.predict(x)
# Betas = np.load("Projeto1/Betas.npy")
for i in range(0, x.shape[1]):
    plt.subplot(2, 5, i+1)
    for j in range(0, x.shape[0]):
        plt.scatter(x[j, i], y[j])
        plt.plot(x[j, i], y_est[j])
    plt.plot(x[:, i], y_est)
    plt.xlabel(f"X{i+1}")
    plt.ylabel("Y")

plt.show()

kf = KFold(n_splits=15, shuffle=False)
# Linear

x = np.load("Projeto1/X_train_regression1.npy")
y = np.load("Projeto1/y_train_regression1.npy")
for idx, idx_teste in kf.split(x):
    y_prever = reg.predict(x[idx_teste])
    print(f"O r^2 do linear é {y_prever-y[idx_teste]} para o teste {idx_teste}")
