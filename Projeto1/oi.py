import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

x = np.load("Projeto1/X_train_regression1.npy")
y = np.load("Projeto1/y_train_regression1.npy")
kf = KFold(n_splits=15, shuffle=False)  # Para o cross Validation
SSE2 = 0
SSE1 = 0
SSE3 = 0
SSE5 = np.array([])
alphas = np.array([])
""" for j in np.arange(0.1, 8, 0.01):
    ridge = linear_model.Ridge(alpha=j)
    SSE2 = 0
    for idx_train, idx_teste in kf.split(x):
        ridge.fit(x[idx_train], y[idx_train])
        y_prever2 = ridge.predict(x[idx_teste])
        SSE = np.linalg.norm(y[idx_teste]-y_prever2)**2
        SSE2 += SSE
        print("Ridge: ", np.linalg.norm(y[idx_teste]-y_prever2)**2)
    print("\n\nMédia Ridge: ", SSE2/kf.get_n_splits())
    q = np.append(q, j)
    SSE5 = np.append(SSE5, SSE2/kf.get_n_splits())
plt.plot(q, SSE5)
print(SSE5.min())
print(q[SSE5.argmin()])
plt.show() """

for j in np.arange(0, 1, 0.0001):
    lasso = linear_model.Lasso(alpha=j)
    SSE3 = 0
    for idx_train, idx_teste in kf.split(x):
        lasso.fit(x[idx_train], y[idx_train])
        y_prever3 = lasso.predict(x[idx_teste])
        SSE = np.linalg.norm(y[idx_teste]-y_prever3)**2
        SSE3 += SSE
        """ print("Lasso: ", np.linalg.norm(y[idx_teste]-y_prever3)**2)  """
    print("\nMédia Lasso: ", SSE3/kf.get_n_splits())
    alphas = np.append(alphas, j)
    SSE5 = np.append(SSE5, SSE3/kf.get_n_splits())
np.save("Projeto1/SSE_lasso.npy", SSE5)
np.save("Projeto1/alphas_lasso.npy", alphas)
plt.plot(alphas, SSE5)
print(SSE5.min())
print(alphas[SSE5.argmin()])
plt.show()


""" reg = linear_model.LinearRegression()
for idx_train, idx_teste in kf.split(x):
    reg.fit(x[idx_train], y[idx_train])
    y_prever1 = reg.predict(x[idx_teste])
    SSE1 += np.linalg.norm(y[idx_teste]-y_prever1)**2
    print("Linear: ", np.linalg.norm(y[idx_teste]-y_prever1)**2)
print("\n\nMédia Linear: ", SSE1/kf.get_n_splits(), "\t\t", kf.get_n_splits()) """


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
