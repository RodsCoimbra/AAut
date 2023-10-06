import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import KFold

# Load das coisas
""" x = np.load("Projeto1_Parte1/Dados_enunciado/X_train_regression1.npy")
y = np.load("Projeto1_Parte1/Dados_enunciado/y_train_regression1.npy") """
x = np.load("Projeto1_Parte2/Dados/X2_train_alpha_regression2.npy")
y = np.load("Projeto1_Parte2/Dados/y2_train_alpha_regression2.npy")
rand = np.random.randint(0, 100000)
# Para o cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=rand)
SSE_Ridge = 0
SSE_Linear = 0
SSE_Lasso = 0

# Variáveis a definir
step = 0.005
max_lasso = 3
max_ridge = 4
prints = False
Load_dados = False
save_dados = False


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

# Linear
reg = linear_model.LinearRegression()
for idx_train, idx_teste in kf.split(x):
    reg.fit(x[idx_train], y[idx_train])
    y_prever1 = reg.predict(x[idx_teste])
    SSE_Linear += np.linalg.norm(y[idx_teste]-y_prever1)**2/(idx_teste.size)
    if (prints == True):
        print("Linear: ", np.linalg.norm(y[idx_teste]-y_prever1)**2)
print("Média Linear: ", SSE_Linear/kf.get_n_splits())
