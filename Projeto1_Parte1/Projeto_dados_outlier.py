import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


x = np.load("Dados_enunciado/X_train_regression1.npy")
y = np.load("Dados_enunciado/y_train_regression1.npy")

reg = linear_model.Ridge()
parameters = {'alpha':(np.arange(0.01,3,0.01))}
GridSearchCV(reg,parameters, n_jobs=-1, cv=10)
# print(reg.intercept_.shape)
# for i in range(0, x.shape[1]):
#     plt.subplot(2, 5, i+1)
#     for j in range(0, x.shape[0]):
#         plt.scatter(x[j, i], y[j])
#     plt.xlabel(f"X{i+1}")
#     plt.ylabel("Y")

# plt.show()
