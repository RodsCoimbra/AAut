from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sys import exit
x = np.load("Projeto1/X_train_regression1.npy")
y = np.load("Projeto1/y_train_regression1.npy")
rand = np.random.randint(0, 100000)
# Para o cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=rand)
SSE_Ridge = 0
SSE_Linear = 0
SSE_Lasso = 0

# Variáveis a definir
step = 0.05
# step = 0.0001
max_lasso = 3
max_ridge = 10
prints = False
Load_dados = False
save_dados = False
max_net = 2
max_l1_ratio = 0.6

SSE_mean = np.array([])
alphas = np.array([])

# Ridge
if (Load_dados == False):
    for j in np.arange(step, max_ridge, step):
        ridge = linear_model.Ridge(alpha=j)
        SSE_Ridge = 0
        for idx_train, idx_teste in kf.split(x):
            ridge.fit(x[idx_train], y[idx_train])
            y_prever2 = ridge.predict(x[idx_teste])
            SSE = np.linalg.norm(y[idx_teste]-y_prever2)**2/(idx_teste.size)
            SSE_Ridge += SSE
        if (prints == True):
            print("\n\nMédia Ridge: ", SSE_Ridge/kf.get_n_splits())
        alphas = np.append(alphas, j)
        SSE_mean = np.append(SSE_mean, SSE_Ridge/kf.get_n_splits())
    if (save_dados == True):
        np.save("Projeto1/SSE_ridge2.npy", SSE_mean)
        np.save("Projeto1/alphas_ridge2.npy", alphas)
else:
    SSE_mean = np.load("Projeto1/SSE_ridge.npy")
    alphas = np.load("Projeto1/alphas_ridge.npy")
plt.plot(alphas, SSE_mean)
print("Valor de SSE mais baixo do Ridge:", SSE_mean.min())
print("Para este valor de alpha:", alphas[SSE_mean.argmin()])
print()
plt.legend("Ridge")

SSE_mean = np.array([])
alphas = np.array([])
# Lasso
if (Load_dados == False):
    for j in np.arange(step, max_lasso, step):
        lasso = linear_model.Lasso(alpha=j)
        SSE_Lasso = 0
        for idx_train, idx_teste in kf.split(x):
            lasso.fit(x[idx_train], y[idx_train])
            y_prever3 = lasso.predict(x[idx_teste])
            SSE = np.linalg.norm(y[idx_teste]-y_prever3)**2/(idx_teste.size)
            SSE_Lasso += SSE
        if (prints == True):
            print("\nMédia Lasso: ", SSE_Lasso/kf.get_n_splits())
        alphas = np.append(alphas, j)
        SSE_mean = np.append(SSE_mean, SSE_Lasso/kf.get_n_splits())
    if (save_dados == True):
        np.save("Projeto1/SSE_lasso2.npy", SSE_mean)
        np.save("Projeto1/alphas_lasso2.npy", alphas)
else:
    SSE_mean = np.load("Projeto1/SSE_lasso.npy")
    alphas = np.load("Projeto1/alphas_lasso.npy")

plt.plot(alphas, SSE_mean)
print("Valor de SSE mais baixo do Lasso:", SSE_mean.min())
print("Para este valor de alpha:", alphas[SSE_mean.argmin()])
print()
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.legend(["Ridge", "Lasso"])
plt.show()


# ElasticNet
i = 0
SSE_mean = np.array([[]])
alphas = np.array([[]])
l1_ratio = np.array([])
if (Load_dados == False):
    for j in np.arange(step, max_net, step):
        for k in np.arange(step, max_l1_ratio + step, step):
            ElasticNet = linear_model.ElasticNet(alpha=j, l1_ratio=k)
            SSE_ElasticNet = 0
            for idx_train, idx_teste in kf.split(x):
                ElasticNet.fit(x[idx_train], y[idx_train])
                y_prever3 = ElasticNet.predict(x[idx_teste])
                SSE = np.linalg.norm(y[idx_teste]-y_prever3)**2/(idx_teste.size)
                SSE_ElasticNet += SSE
            if (prints == True):
                print("\nMédia ElasticNet: ", SSE_ElasticNet/kf.get_n_splits())
            SSE_mean = np.append(SSE_mean, SSE_ElasticNet/kf.get_n_splits())
            alphas = np.append(alphas, j)
            l1_ratio = np.append(l1_ratio, k)
    if (save_dados == True):
        np.save("Projeto1/SSE_elastic.npy", SSE_mean)
        np.save("Projeto1/alphas_elastic.npy", alphas)
        np.save("Projeto1/l1_ratio_elastic.npy", l1_ratio)
else:
    SSE_mean = np.load("Projeto1/SSE_elastic.npy")
    alphas = np.load("Projeto1/alphas_elastic.npy")
SSE_mean = SSE_mean.reshape(
    round((max_l1_ratio)/step), round((max_net-step)/step))
alphasgrid, l1_ratiogrid = np.meshgrid(
    np.arange(step, max_net, step), np.arange(step, max_l1_ratio + step, step))
fig = plt.figure(figsize=(14, 9))
ax = plt.axes(projection='3d')
my_cmap = plt.get_cmap('gist_earth')
# Creating plot
surface = ax.plot_surface(alphasgrid, l1_ratiogrid, SSE_mean,
                          cmap=my_cmap, edgecolor='none')
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel('Alpha')
ax.set_ylabel('l1_ratio')
ax.set_zlabel('SSE')
ax.set_title('ElasticNet')
print("Valor de SSE mais baixo do ElasticNet:", SSE_mean.min())
print("Para este valor de alpha:",
      alphas[SSE_mean.argmin()], "e l1_ratio:", l1_ratio[SSE_mean.argmin()])
print()
plt.show()

# Linear
reg = linear_model.LinearRegression()
for idx_train, idx_teste in kf.split(x):
    reg.fit(x[idx_train], y[idx_train])
    y_prever1 = reg.predict(x[idx_teste])
    SSE_Linear += np.linalg.norm(y[idx_teste]-y_prever1)**2/(idx_teste.size)
    if (prints == True):
        print("Linear: ", np.linalg.norm(y[idx_teste]-y_prever1)**2)
print("Média Linear: ", SSE_Linear/kf.get_n_splits())


""" x_train, x_teste, y_train, y_teste = train_test_split(x, y, shuffle=True, test_size=1)
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
print("Lasso: ", np.linalg.norm(y_teste-y_prever3)**2) """
