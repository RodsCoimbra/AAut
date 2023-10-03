import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import KFold

# Load das coisas
x = np.load("Projeto1_Parte2/Dados/X_train_alpha_regression2.npy")
y = np.load("Projeto1_Parte2/Dados/y_train_alpha_regression2.npy")
x2 = np.load("Projeto1_Parte2/Dados/X2_train_alpha_regression2.npy")
y2 = np.load("Projeto1_Parte2/Dados/y2_train_alpha_regression2.npy")
rand = np.random.randint(0, 100000)
# Para o cross Validation
""" kf = KFold(n_splits=y.size, shuffle=True, random_state=rand) """
kf = KFold(n_splits=10, shuffle=True, random_state=rand)
SSE_Linear = 0
prints = False

# Linear
reg = linear_model.LinearRegression()
for idx_train, idx_teste in kf.split(x):
    reg.fit(x[idx_train], y[idx_train])
    y_prever1 = reg.predict(x[idx_teste])
    SSE_Linear += np.linalg.norm(y[idx_teste]-y_prever1)**2

print("Média Linear1: ", SSE_Linear/kf.get_n_splits())

SSE_Linear = 0
for idx_train, idx_teste in kf.split(x2):
    reg.fit(x2[idx_train], y2[idx_train])
    y_prever2 = reg.predict(x2[idx_teste])
    SSE_Linear += np.linalg.norm(y2[idx_teste]-y_prever2)**2

print("Média Linear2: ", SSE_Linear/kf.get_n_splits())
