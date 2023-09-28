import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import KFold
from sys import exit
x = np.load("Projeto1/X_train_regression1.npy")
y = np.load("Projeto1/y_train_regression1.npy")
SSE_Ridge = 0

# Variáveis a definir
step = 0.05
runs = 300
# step = 0.0001
max_ridge = 5
prints = False
Load_dados = False
save_dados = False


SSE_total = np.zeros(
    np.arange(step, max_ridge, step).shape, dtype=np.longdouble)
# Ridge
if (Load_dados == False):
    for i in range(runs):
        print("Run: ", i)
        rand = np.random.randint(0, 100000)
        # Para o cross Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=rand)
        SSE_mean = np.array([])
        alphas = np.array([])
        for k, j in enumerate(np.arange(step, max_ridge, step)):
            ridge = linear_model.Ridge(alpha=j)
            SSE_Ridge = 0
            for idx_train, idx_teste in kf.split(x):
                ridge.fit(x[idx_train], y[idx_train])
                y_prever2 = ridge.predict(x[idx_teste])
                SSE = np.linalg.norm(
                    y[idx_teste]-y_prever2)**2/(idx_teste.size)
                SSE_Ridge += SSE
            if (prints == True):
                print("\n\nMédia Ridge: ", SSE_Ridge/kf.get_n_splits())
            alphas = np.append(alphas, j)
            SSE_mean = np.append(SSE_mean, SSE_Ridge/kf.get_n_splits())
            SSE_total[k] += SSE_mean[k]
    if (save_dados == True):
        np.save("Projeto1/alphas_ridge_melhor.npy", alphas)
        np.save("Projeto1/SSE_ridge_melhor.npy", SSE_total)
else:
    SSE_true = np.load("Projeto1/SSE_ridge_melhor.npy")
    alphas = np.load("Projeto1/alphas_ridge.npy")

SSE_total = SSE_total/runs
plt.plot(alphas, SSE_total, label="Ridge")
print("Valor de SSE mais baixo do Ridge:", SSE_total.min())
print("Para este valor de alpha:", alphas[SSE_total.argmin()])
print()
plt.xlabel("Alpha")
plt.ylabel("SSE")
plt.legend(["Ridge"])
plt.show()
