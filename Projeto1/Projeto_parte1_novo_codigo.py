import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split


""" x = np.load("Projeto1/X_train_regression1.npy")
y = np.load("Projeto1/y_train_regression1.npy")
alpha_lasso = np.zeros([150, 2])
alpha_ridge = np.zeros([150, 2])
for j in range(0, 5000):
    x_train, x_teste, y_train, y_teste = train_test_split(
        x, y, shuffle=True, test_size=5)
    # Ridge

    r2 = 9999
    for i in np.arange(0.1, 15.1, 0.1):
        ridge = linear_model.Ridge(alpha=i)
        ridge.fit(x_train, y_train)
        y_prever = ridge.predict(x_teste)
        a = np.linalg.norm(y_teste-y_prever)**2
        alpha_ridge[round(i*10-1), 0] += a
        if (a < r2):
            r2 = a
            k = i
    alpha_ridge[round(k*10-1), 1] += 1

    # Lasso
    r2 = 9999
    for i in np.arange(0.1, 15.1, 0.1):
        lasso = linear_model.Lasso(alpha=i)
        lasso.fit(x_train, y_train)
        y_prever = lasso.predict(x_teste)
        a = np.linalg.norm(y_teste-y_prever)**2
        alpha_lasso[round(i*10-1), 0] += a
        if (a < r2):
            r2 = a
            k = i
    alpha_lasso[round(k*10-1), 1] += 1
np.save("Projeto1/alpha_ridgeSSE.npy", alpha_ridge)
np.save("Projeto1/alpha_lassoSSE.npy", alpha_lasso)
alpha_ridge[:, 0] = alpha_ridge[:, 0]/5000
alpha_lasso[:, 0] = alpha_lasso[:, 0]/5000 """
alpha_ridge = np.load("Projeto1/alpha_ridgeSSE.npy")
alpha_lasso = np.load("Projeto1/alpha_lassoSSE.npy")
alpha_ridge[:, 0] = alpha_ridge[:, 0]/5000
alpha_lasso[:, 0] = alpha_lasso[:, 0]/5000
alphas = np.arange(0.1, 15.1, 0.1)
plt.subplot(2, 1, 1)
plt.plot(alphas, alpha_ridge[:, 0], label="Ridge")
plt.grid()
plt.title("Ridge")
plt.subplot(2, 1, 2)
plt.plot(alphas, alpha_lasso[:, 0], label="Lasso")
plt.grid()
plt.title("Lasso")
plt.show()
print(alpha_ridge[:, 1], "\n\n", alpha_lasso[:, 1], "\n")
print("Melhor alpha do ridge: ", (alpha_ridge[:, 1].argmax()+1)/10,
      "\t Melhor alpha do Lasso", (alpha_lasso[:, 1].argmax()+1)/10)
