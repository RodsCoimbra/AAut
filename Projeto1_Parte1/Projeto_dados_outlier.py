import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


x = np.load("Dados_enunciado/X_train_regression1.npy")
y = np.load("Dados_enunciado/y_train_regression1.npy")

reg = linear_model.LinearRegression()
CV = [3, 5, 10, 15]
for i in CV:
    kf = KFold(n_splits=i, shuffle=True)
    MSE = 0
    for idx_train, idx_teste in kf.split(x):
        reg.fit(x[idx_train], y[idx_train])
        y_prever1 = reg.predict(x[idx_teste])
        MSE += np.linalg.norm(y[idx_teste]-y_prever1)**2/(idx_teste.size)
    print(MSE/kf.get_n_splits())
    # grid = GridSearchCV(reg, parameters,n_jobs=-1, cv=i, scoring='neg_mean_squared_error')
    # print(grid.cv_results_['mean_test_score'])
# # print(reg.intercept_.shape)
# # for i in range(0, x.shape[1]):
# #     plt.subplot(2, 5, i+1)
# #     for j in range(0, x.shape[0]):
# #         plt.scatter(x[j, i], y[j])
# #     plt.xlabel(f"X{i+1}")
# #     plt.ylabel("Y")

# # plt.show()
