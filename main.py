import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import array
from sklearn.preprocessing import StandardScaler


""" x = np.load("Proj1_Parte2/Dados/X_train_regression2.npy")
y = np.load("Proj1_Parte2/Dados/y_train_regression2.npy") """
x = np.load("x2.npy")
y = np.load("y2.npy")

scalerx = StandardScaler().fit(x)
x = scalerx.transform(x)
scalery = StandardScaler().fit(y)
y = scalery.transform(y)

Lin = LinearRegression()
Indx = np.zeros(5)
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
        """ a = np.linalg.norm(y_previsto - y)**2 """
        a = 0
        for [y_prev, y_real] in zip(y_previsto, y):
            if (abs(y_prev - y_real) < 0.5):
                a += 1
        if (a > Max):
            Indx = [a, idx-1, idx2, j, j2]
            Max = a
print(Indx)
Lin.fit([x[Indx[1], :], x[Indx[2], :]], [y[Indx[1]], y[Indx[2]]])
y_previsto = Lin.predict(x)
for idx, [y_prev, y_real] in enumerate(zip(y_previsto, y)):
    if (abs(y_prev - y_real) < 0.5):
        save = np.append(save, x[idx])
        savey = np.append(savey, y_real)
    else:
        save2 = np.append(save2, x[idx])
        savey2 = np.append(savey2, y_real)
    Min[idx] = abs(y_prev - y_real)

""" save = save.reshape(round(save.shape[0]/4), 4)
save2 = save2.reshape(round(save2.shape[0]/4), 4)
savey = savey.reshape(-1, 1)
savey2 = savey2.reshape(-1, 1)
np.save("x1.npy", save)
np.save("x2.npy", save2)
np.save("y1.npy", savey)
np.save("y2.npy", savey2) """
plt.scatter(np.arange(0, y.ravel().size), Min)
plt.show()
