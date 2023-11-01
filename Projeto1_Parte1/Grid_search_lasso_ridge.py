import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV


x = np.load("Dados_enunciado/X_train_regression1.npy")
y = np.load("Dados_enunciado/y_train_regression1.npy")
CV = [3, 5, 15]
for i in CV:
    plt.figure()
    reg = linear_model.Ridge()
    parameters = {'alpha': (np.arange(0.01, 5, 0.001))}
    grid = GridSearchCV(reg, parameters, n_jobs=-1, cv=i,
                        scoring='neg_mean_squared_error')
    grid.fit(x, y)
    grafico = grid.cv_results_
    print("Ridge alpha ->", grid.best_params_['alpha'])
    print("Score ->", abs(grid.best_score_))
    x1 = grafico['param_alpha']
    y1 = abs(grafico['mean_test_score'])
    reg2 = linear_model.Lasso()
    parameters2 = {'alpha': (np.arange(0.01, 5, 0.001))}
    grid2 = GridSearchCV(reg2, parameters2, n_jobs=-1,
                         cv=i, scoring='neg_mean_squared_error')
    grid2.fit(x, y)
    grafico2 = grid2.cv_results_
    print("\nLasso alpha ->", grid2.best_params_['alpha'])
    print("Score ->", abs(grid2.best_score_))
    x2 = grafico2['param_alpha']
    y2 = abs(grafico2['mean_test_score'])
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.title(f'CV = {i}')
    plt.scatter(grid.best_params_['alpha'], abs(grid.best_score_), color='red')
    plt.scatter(grid2.best_params_['alpha'], abs(
        grid2.best_score_), color='green')
    plt.xlabel('Alpha')
    plt.ylabel('MSE')
    plt.legend(['Ridge', 'Lasso'])
plt.show()
