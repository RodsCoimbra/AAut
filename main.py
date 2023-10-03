import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import KFold


x = np.load("Projeto1_Parte2/Dados/X_train_regression2.npy")
y = np.load("Projeto1_Parte2/Dados/y_train_regression2.npy")
""" x = np.load("x2.npy")
y = np.load("y2.npy") """

scalerx = StandardScaler().fit(x)
x = scalerx.transform(x)
scalery = StandardScaler().fit(y)
y = scalery.transform(y)

Lin = LinearRegression()
SSE_Min = np.full([2], 10000000000)
Graficox = []
Graficoy = []
for Valor in np.arange(0.575, 0.588, 0.0005):
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
            """ a = np.linalg.norm(y_previsto - y)**2 """
            a = 0
            for [y_prev, y_real] in zip(y_previsto, y):
                if (abs(y_prev - y_real) < Valor):
                    a += 1
            if (a > Max):
                Indx = [a, idx-1, idx2]
                Max = a
    if (Indx[0] == 0):
        continue
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
    if (len(savey) < 10 or len(savey2) < 10):
        continue
    save = save.reshape(round(save.shape[0]/4), 4)
    save2 = save2.reshape(round(save2.shape[0]/4), 4)
    savey = savey.reshape(-1, 1)
    savey2 = savey2.reshape(-1, 1)
    save = scalerx.inverse_transform(save)
    savey = scalery.inverse_transform(savey)
    save2 = scalerx.inverse_transform(save2)
    savey2 = scalery.inverse_transform(savey2)
    # Linear
    reg = linear_model.LinearRegression()
    SSE_Linear1 = 0
    SSE_Linear2 = 0
    for a in range(0, 5):
        rand = np.random.randint(0, 100000)
        # Para o cross Validation
        kf = KFold(n_splits=10, shuffle=True, random_state=rand)
        for idx_train, idx_teste in kf.split(save):
            reg.fit(save[idx_train], savey[idx_train])
            y_prever1 = reg.predict(save[idx_teste])
            SSE_Linear1 += np.linalg.norm(savey[idx_teste] -
                                          y_prever1)**2
        for idx_train, idx_teste in kf.split(save2):
            reg.fit(save2[idx_train], savey2[idx_train])
            y_prever1 = reg.predict(save2[idx_teste])
            SSE_Linear2 += np.linalg.norm(savey2[idx_teste] -
                                          y_prever1)**2
    SSE_Linear1 = SSE_Linear1/5
    SSE_Linear2 = SSE_Linear2/5
    Final = np.linalg.norm(
        [SSE_Linear1/kf.get_n_splits(), SSE_Linear2/kf.get_n_splits()])
    Graficox = np.append(Graficox, Valor)
    Graficoy = np.append(Graficoy, Final)
    if (SSE_Min[0] > Final):
        SSE_Min = [Final, Valor]
    print("\nPara o valor de ", Valor,
          " a norma da média dos dois SSE é: ", Final)
    print(f"Referência: {SSE_Linear1:.4f}\t {SSE_Linear2:.4f} \nNumero de pontos 1: {Indx[0]} \t Numero de pontos 2: {save2.shape[0]}")
print("\n\n\nO final de todos foi: ",
      SSE_Min[0], " para o valor de ", SSE_Min[1])
plt.plot(Graficox, Graficoy)
plt.show()
""" np.save("Projeto1_Parte2/Dados/X_train_alpha_regression2.npy", save)
np.save("Projeto1_Parte2/Dados/X2_train_alpha_regression2.npy", save2)
np.save("Projeto1_Parte2/Dados/y_train_alpha_regression2.npy", savey)
np.save("Projeto1_Parte2/Dados/y2_train_alpha_regression2.npy", savey2) """
""" plt.scatter(np.arange(0, y.ravel().size), Min)
plt.show() """
