import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

x = np.load("Projeto1/X_train_regression1.npy")
y = np.load("Projeto1/y_train_regression1.npy")
kf = KFold(n_splits=15, shuffle=False)  # Para o cross Validation

#Variáveis a definir
SSE_Ridge = 0
SSE_Linear = 0
SSE_Lasso = 0
#step = 0.0001
step = 0.1
max_lasso = 1
max_ridge = 3


SSE_mean = np.array([])
alphas = np.array([])


# Ridge
for j in np.arange(0, max_ridge, step):
    ridge = linear_model.Ridge(alpha=j)
    SSE_Ridge = 0
    for idx_train, idx_teste in kf.split(x):
        ridge.fit(x[idx_train], y[idx_train])
        y_prever2 = ridge.predict(x[idx_teste])
        SSE = np.linalg.norm(y[idx_teste]-y_prever2)**2
        SSE_Ridge += SSE
        print("Ridge: ", np.linalg.norm(y[idx_teste]-y_prever2)**2)
    print("\n\nMédia Ridge: ", SSE_Ridge/kf.get_n_splits())
    alphas = np.append(alphas, j)
    SSE_mean = np.append(SSE_mean, SSE_Ridge/kf.get_n_splits())
np.save("Projeto1/SSE_ridge.npy", SSE_mean)
np.save("Projeto1/alphas_ridge.npy", alphas)
plt.plot(alphas, SSE_mean)
print("Valor de SSE mais baixo do Ridge: ", SSE_mean.min())
print("Para este valor de alpha", alphas[SSE_mean.argmin()])
plt.show()



SSE_mean = np.array([])
alphas = np.array([])
#Lasso
for j in np.arange(0, max_lasso, step):
    lasso = linear_model.Lasso(alpha=j)
    SSE_Lasso = 0
    for idx_train, idx_teste in kf.split(x):
        lasso.fit(x[idx_train], y[idx_train])
        y_prever3 = lasso.predict(x[idx_teste])
        SSE = np.linalg.norm(y[idx_teste]-y_prever3)**2
        SSE_Lasso += SSE
    print("\nMédia Lasso: ", SSE_Lasso/kf.get_n_splits())
    alphas = np.append(alphas, j)
    SSE_mean = np.append(SSE_mean, SSE_Lasso/kf.get_n_splits())
np.save("Projeto1/SSE_lasso.npy", SSE_mean)
np.save("Projeto1/alphas_lasso.npy", alphas)
plt.plot(alphas, SSE_mean)
print(SSE_mean.min())
print(alphas[SSE_mean.argmin()])
plt.show()

#Linear
reg = linear_model.LinearRegression()
for idx_train, idx_teste in kf.split(x):
    reg.fit(x[idx_train], y[idx_train])
    y_prever1 = reg.predict(x[idx_teste])
    SSE_Linear += np.linalg.norm(y[idx_teste]-y_prever1)**2
    print("Linear: ", np.linalg.norm(y[idx_teste]-y_prever1)**2)
print("\n\nMédia Linear: ", SSE_Linear/kf.get_n_splits(), "\t\t", kf.get_n_splits())


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
