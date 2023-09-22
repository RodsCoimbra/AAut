import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

# Método linear
x = np.load("Projeto1/X_train_regression1.npy")
y = np.load("Projeto1/y_train_regression1.npy")
x1 = np.append(np.ones((x.shape[0], 1)), x, axis=1)
Betas = (np.linalg.inv(x1.T @ x1) @ x1.T) @ y
""" print("Os valores dos Betas são:\n", Betas.T) """
SSE = np.linalg.norm(y-x1@Betas)**2
SSE2 = (y - y.mean()).T @ (y - y.mean())
""" SSE3 = (y - x @ Betas).T @ (y - x @ Betas)
SSE4 = ((y-y_avg)**2).sum() """  # Duas maneiras extras de computar o SSE e o SSE2, não sei qual a mais rápida
""" print()
print("O r^2 é ", 1 - (SSE/SSE2)[0, 0]) """


# Scikit-learn Método Linear
""" x_train, x_teste, y_train, y_teste = train_test_split(x, y, shuffle=True, test_size=4) """
x_train = x
y_train = y
total = np.zeros(2)
kf = KFold(n_splits=3, shuffle=True)  # Para o cross Validation

#Linear
reg = linear_model.LinearRegression()
for idx_train, idx_teste in kf.split(x_train):
    reg.fit(x_train[idx_train], y_train[idx_train])
    y_prever = reg.predict(x_train[idx_teste])
    r2 = r2_score(y_true=y_train[idx_teste], y_pred=y_prever)
    total[0] += r2
"""     print(f"O r^2 do linear é {r2:.4f}") """
print(f"A média é {total[0]/kf.get_n_splits():.4f} para o Linear \n")

#Ridge
total[0] = 0
for i in np.arange(0.1, 1.1, 0.1):
    ridge = linear_model.Ridge(alpha=i)
    for idx_train, idx_teste in kf.split(x_train):
        ridge.fit(x_train[idx_train], y_train[idx_train])
        y_prever = ridge.predict(x_train[idx_teste])
        r2 = r2_score(y_true=y_train[idx_teste], y_pred=y_prever)
        if(total[0] < r2):
            total[0] = r2
            total[1] = i
        """ total[1] += r2
    print(f"A média é {total[1]/kf.get_n_splits():.4f} para o Ridge com alpha = {i}\n")
    total[1] = 0 """
print(f"O melhor para o Ridge foi {total[0]:.4f} com alpha = {total[1]}\n")

#Lasso 
total = np.zeros(2)
for i in np.arange(0.1, 1.1, 0.1):
    lasso = linear_model.Lasso(alpha=i)
    for idx_train, idx_teste in kf.split(x_train):
        lasso.fit(x_train[idx_train], y_train[idx_train])
        y_prever = lasso.predict(x_train[idx_teste])
        r2 = r2_score(y_true=y_train[idx_teste], y_pred=y_prever)
        if(total[0] < r2):
            total[0] = r2
            total[1] = i
        """ total[1] += r2
    print(f"A média é {total[1]/kf.get_n_splits():.4f} para o Lasso com alpha = {i}\n")
    total[1] = 0 """
        
print(f"O melhor para o Lasso foi {total[0]:.4f} com alpha = {total[1]}\n")


""" y_prever = reg.predict(x_teste)
r2 = r2_score(y_true=y_teste, y_pred=y_prever)
print(f"O r^2 final é {r2:.4f}") """
