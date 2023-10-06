import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold

# Load das coisas
x1 = np.load("Projeto1_Parte2/Dados/X_train_alpha_regression2.npy")
y1 = np.load("Projeto1_Parte2/Dados/y_train_alpha_regression2.npy")
x2 = np.load("Projeto1_Parte2/Dados/X2_train_alpha_regression2.npy")
y2 = np.load("Projeto1_Parte2/Dados/y2_train_alpha_regression2.npy")
x = np.load("Projeto1_Parte2/Dados/X_train_regression2.npy")
y = np.load("Projeto1_Parte2/Dados/y_train_regression2.npy")

rand = np.random.randint(0, 100000)
# Para o cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=rand)
SSE_soma = 0
# Linear
reg1 = linear_model.LinearRegression()
reg2 = linear_model.LinearRegression()
for [(idx_train, idx_teste), (idx_train2, idx_teste2)] in zip(kf.split(x1), kf.split(x2)):
    SSE_final_array = []
    reg1.fit(x1[idx_train], y1[idx_train])
    reg2.fit(x2[idx_train2], y2[idx_train2])
    y_prever1 = reg1.predict(x)
    y_prever2 = reg2.predict(x)
    SSE_final_array = np.where((y - y_prever1)**2 > (y-y_prever2)**2,
                               (y-y_prever2)**2, (y - y_prever1)**2)
    SSE_soma += SSE_final_array.sum()

print("\nMédia Linear SSE: ", SSE_soma/kf.get_n_splits())
SSE_final_array = []
reg1.fit(x, y)
y_prever3 = reg1.predict(x)
SSE = np.linalg.norm(y-y_prever3)**2
print("\nE agora SSE", SSE)
