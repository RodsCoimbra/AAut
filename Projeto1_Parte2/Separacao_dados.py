import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import KFold


x = np.load("Projeto1_Parte2/Dados/X_train_regression2.npy")
y = np.load("Projeto1_Parte2/Dados/y_train_regression2.npy")

scalerx = StandardScaler().fit(x)
x = scalerx.transform(x)
scalery = StandardScaler().fit(y)
y = scalery.transform(y)
Valor = 0.375
Lin = LinearRegression()
Indx = np.zeros(3)
Max = 0
Min = np.zeros(y.ravel().size)
save = []
save2 = []
savey = []
savey2 = []
for idx, [i, j] in enumerate(zip(x, y), 1):
    for idx2, [i2, j2] in enumerate(zip(x[idx:, :], y[idx:]), idx):
        Lin.fit([i, i2], [j, j2])
        y_previsto = Lin.predict(x)
        a = 0
        for [y_prev, y_real] in zip(y_previsto, y):
            if (abs(y_prev - y_real) < Valor):
                a += 1
        if (a > Max):
            Indx = [a, idx-1, idx2]
            Max = a
Lin.fit([x[Indx[1], :], x[Indx[2], :]], [y[Indx[1]], y[Indx[2]]])
y_previsto = Lin.predict(x)
for idx, [y_prev, y_real] in enumerate(zip(y_previsto, y)):
    if (abs(y_prev - y_real) < Valor):
        save = np.append(save, x[idx])
        savey = np.append(savey, y_real)
    else:
        save2 = np.append(save2, x[idx])
        savey2 = np.append(savey2, y_real)
    Min[idx] = abs(y_prev - y_real)
save = save.reshape(round(save.shape[0]/4), 4)
save2 = save2.reshape(round(save2.shape[0]/4), 4)
savey = savey.reshape(-1, 1)
savey2 = savey2.reshape(-1, 1)
save = scalerx.inverse_transform(save)
savey = scalery.inverse_transform(savey)
save2 = scalerx.inverse_transform(save2)
savey2 = scalery.inverse_transform(savey2)
""" 
np.save("Projeto1_Parte2/Dados/X_train_alpha_regression2.npy", save)
np.save("Projeto1_Parte2/Dados/X2_train_alpha_regression2.npy", save2)
np.save("Projeto1_Parte2/Dados/y_train_alpha_regression2.npy", savey)
np.save("Projeto1_Parte2/Dados/y2_train_alpha_regression2.npy", savey2)
 """
