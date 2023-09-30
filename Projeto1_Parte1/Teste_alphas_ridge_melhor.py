import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import KFold

# Load das coisas
x = np.load("Projeto1/X_train_regression1.npy")
y = np.load("Projeto1/y_train_regression1.npy")
SSE_Ridge = 0

# Variáveis a definir
step = 0.01
runs = 5
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
        kf = KFold(n_splits=3, shuffle=True, random_state=rand)
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
        np.save("Projeto1/alphas_ridge_melhor_fold3.npy", alphas)
        np.save("Projeto1/SSE_ridge_melhor_fold3.npy", SSE_total/runs)
    SSE_total = SSE_total/runs
else:
    SSE_total_5 = np.load("Projeto1/SSE_ridge_melhor.npy")
    alphas = np.load("Projeto1/alphas_ridge_melhor.npy")
    SSE_total_3 = np.load("Projeto1/SSE_ridge_melhor_fold3.npy")
    SSE_total_15 = np.load("Projeto1/SSE_ridge_melhor_fold15.npy")
    plt.plot(alphas, SSE_total_5, label="Ridge 5")
    plt.plot(alphas, SSE_total_3, label="Ridge 3")
    plt.plot(alphas, SSE_total_15, label="Ridge 15")
    print("Valor de SSE mais baixo do Ridge:", SSE_total_3.min())
    print("Para este valor de alpha:", alphas[SSE_total_3.argmin()])
    print("Valor de SSE mais baixo do Ridge:", SSE_total_5.min())
    print("Para este valor de alpha:", alphas[SSE_total_5.argmin()])
    print("Valor de SSE mais baixo do Ridge:", SSE_total_15.min())
    print("Para este valor de alpha:", alphas[SSE_total_15.argmin()])
    SSE_total = (SSE_total_3 + SSE_total_5 + SSE_total_15)/3
print()


print("Valor de SSE mais baixo do Ridge de sempre:", SSE_total.min())
print("Para este valor de alpha:", alphas[SSE_total.argmin()])
plt.plot(alphas, SSE_total, label="Ridge_final")
plt.legend()
plt.xlabel("Alpha")
plt.ylabel("SSE")
plt.show()
