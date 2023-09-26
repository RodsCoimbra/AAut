import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

x = np.load("Projeto1/X_train_regression1.npy")
y = np.load("Projeto1/y_train_regression1.npy")
x_train, x_teste, y_train, y_teste = train_test_split(x, y, shuffle=True, test_size=4)
ridge = linear_model.Ridge(alpha=1.5)
lasso = linear_model.Lasso(alpha=1.5)
reg = linear_model.LinearRegression()

reg.fit(x_train, y_train)
ridge.fit(x_train, y_train)
lasso.fit(x_train, y_train)

y_prever1 = reg.predict(x_teste)
print("Linear: ", np.linalg.norm(y_teste-y_prever1)**2)
y_prever2 = ridge.predict(x_teste)
print("Ridge: ", np.linalg.norm(y_teste-y_prever2)**2)
y_prever3 = lasso.predict(x_teste)
print("Lasso: ", np.linalg.norm(y_teste-y_prever3)**2)