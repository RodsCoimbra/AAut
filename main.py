import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import KFold
from sys import exit


def RANSAC(x, y, inliers_range):
    Max = 0
    Resultados = np.zeros((1, 3))
    for idx, [i, j] in enumerate(zip(x, y), 1):
        for idx2, [i2, j2] in enumerate(zip(x[idx:, :], y[idx:]), idx):
            Lin.fit([i, i2], [j, j2])
            y_previsto = Lin.predict(x)
            Counter = 0
            for [y_prev, y_real] in zip(y_previsto, y):
                if (abs(y_prev - y_real) < inliers_range):
                    Counter += 1
            if (Counter > Max):
                Resultados = np.array([[Counter, idx-1, idx2]])
                Max = Counter
            elif (Counter == Max):
                Resultados = np.vstack((Resultados, [Counter, idx-1, idx2]))
    return Resultados


if __name__ == '__main__':
    x = np.load("Projeto1_Parte2/Dados/X_train_regression2.npy")
    y = np.load("Projeto1_Parte2/Dados/y_train_regression2.npy")

    # Normalizar os dados
    scalerx = StandardScaler().fit(x)
    x = scalerx.transform(x)
    scalery = StandardScaler().fit(y)
    y = scalery.transform(y)

    Lin = LinearRegression()
    SSE_Min = np.full([2], 10000000000)
    Graficox = []
    Graficoy = []
    for Valor in np.arange(0.03, 0.9, 0.01):
        Indx = RANSAC(x, y, Valor)
        # Caso em que não há inliers
        if (Indx[0, 0] == 0):
            continue
        # Separação dados
        for resultados in Indx:
            Best_Line = np.full([4, 1], 100000000)
            save = []
            save2 = []
            savey = []
            savey2 = []
            Lin.fit([x[resultados[1], :], x[resultados[2], :]],
                    [y[resultados[1]], y[resultados[2]]])
            y_previsto = Lin.predict(x)
            for idx, [y_prev, y_real] in enumerate(zip(y_previsto, y)):
                if (abs(y_prev - y_real) < Valor):
                    save = np.append(save, x[idx])
                    savey = np.append(savey, y_real)
                else:
                    save2 = np.append(save2, x[idx])
                    savey2 = np.append(savey2, y_real)
            # Um deles tem menos que 10 pontos, logo não é possível fazer 10 folds
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
            reg1 = linear_model.LinearRegression()
            reg2 = linear_model.LinearRegression()
            SSE_final = 0
            media = 5
            for a in range(0, media):
                rand = np.random.randint(0, 100000)
                # Para o cross Validation
                kf = KFold(n_splits=10, shuffle=True, random_state=rand)
                for [(idx_train, idx_teste), (idx_train2, idx_teste2)] in zip(kf.split(save), kf.split(save2)):
                    reg1.fit(save[idx_train], savey[idx_train])
                    y_prever1 = reg1.predict(x)
                    reg2.fit(save2[idx_train2], savey2[idx_train2])
                    y_prever2 = reg2.predict(x)
                    SSE = np.where((y - y_prever1)**2 > (y-y_prever2)**2,
                                   (y-y_prever2)**2, (y - y_prever1)**2)  # Guardar o melhor SE para cada ponto
                    SSE_final += np.sum(SSE)
            if (Best_Line[0] > SSE_final):
                Best_Line = [SSE_final, resultados[0],
                             resultados[1], resultados[2]]
                savefinal = save
                saveyfinal = savey
                save2final = save2
                savey2final = savey2

        save = savefinal
        savey = saveyfinal
        save2 = save2final
        savey2 = savey2final
        SSE_final = Best_Line[0]/(media*kf.get_n_splits())
        Graficox = np.append(Graficox, Valor)
        Graficoy = np.append(Graficoy, SSE_final)
        if (SSE_Min[0] > SSE_final):
            SSE_Min = [SSE_final, Valor]
        print("\nPara o valor de ", Valor,
              " o SSE é: ", SSE_final)
        print(
            f"Referência: Reta 1 entre os pontos {Best_Line[2]}-{Best_Line[3]}\nNumero de pontos 1 = {Best_Line[1]} \t Numero de pontos 2 = {save2.shape[0]}")
    print("\n\n\nO final de todos foi: ",
          SSE_Min[0], " para o valor de ", SSE_Min[1])
    plt.plot(Graficox, Graficoy)
    plt.show()
    """ np.save("Projeto1_Parte2/Dados/X_train_alpha_regression2.npy", save)
    np.save("Projeto1_Parte2/Dados/X2_train_alpha_regression2.npy", save2)
    np.save("Projeto1_Parte2/Dados/y_train_alpha_regression2.npy", savey)
    np.save("Projeto1_Parte2/Dados/y2_train_alpha_regression2.npy", savey2) """
