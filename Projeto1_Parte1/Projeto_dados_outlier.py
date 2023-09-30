import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


x = np.load("Projeto1/X_train_regression1.npy")
y = np.load("Projeto1/y_train_regression1.npy")
reg = linear_model.LinearRegression()
reg.fit(x, y)
for i in range(0, x.shape[1]):
    plt.subplot(2, 5, i+1)
    for j in range(0, x.shape[0]):
        plt.scatter(x[j, i], y[j])
    plt.xlabel(f"X{i+1}")
    plt.ylabel("Y")

plt.show()

