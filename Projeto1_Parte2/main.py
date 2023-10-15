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
    Lin = LinearRegression()
    for idx, [i, j] in enumerate(zip(x, y), 1):
        for idx2, [i2, j2] in enumerate(zip(x[idx:, :], y[idx:]), idx):
            Lin.fit([i, i2], [j, j2])
            y_previsto = Lin.predict(x)
            Counter = 0
            for [y_prev, y_real] in zip(y_previsto, y):
                if (abs(y_prev - y_real) <= inliers_range):
                    Counter += 1
            if (Counter > Max):
                Resultados = np.array([[Counter, idx-1, idx2]])
                Max = Counter
            elif (Counter == Max):
                Resultados = np.vstack((Resultados, [Counter, idx-1, idx2]))
    return Resultados


def Tratamento_dados(save, save2, savey, savey2, scalex, scaley):
    save = save.reshape(round(save.shape[0]/4), 4)
    save2 = save2.reshape(round(save2.shape[0]/4), 4)
    savey = savey.reshape(-1, 1)
    savey2 = savey2.reshape(-1, 1)
    save = scalex.inverse_transform(save)
    savey = scaley.inverse_transform(savey)
    save2 = scalex.inverse_transform(save2)
    savey2 = scaley.inverse_transform(savey2)
    return save, save2, savey, savey2


if __name__ == '__main__':
    x = np.load("Projeto1_Parte2/Dados/X_train_regression2.npy")
    y = np.load("Projeto1_Parte2/Dados/y_train_regression2.npy")

    # Normalizar os dados
    scalerx = StandardScaler().fit(x)
    x_scaler = scalerx.transform(x)
    scalery = StandardScaler().fit(y)
    y_scaler = scalery.transform(y)
    SSE_Min = np.full([2], 10000000000)
    Graficox = []
    Graficoy = []
    for Valor in np.arange(0.01, 1, 0.01):
        Indx = RANSAC(x_scaler, y_scaler, Valor)
        # Caso em que um deles tem menos que 10 pontos, logo não é possível fazer 10 folds
        if (Indx[0, 0] < 10 or Indx[0, 0] > 90):
            print(Valor, " não tem 10 pontos em cada reta")
            continue
        # Separação dados
        Best_Line = np.full([4, 1], 100000000)
        for resultados in Indx:
            dados = []
            dados2 = []
            dadosy = []
            dadosy2 = []
            linear = LinearRegression()
            linear.fit([x_scaler[resultados[1], :], x_scaler[resultados[2], :]],
                       [y_scaler[resultados[1]], y_scaler[resultados[2]]])
            y_previsto = linear.predict(x_scaler)
            for idx, [y_prev, y_real] in enumerate(zip(y_previsto, y_scaler)):
                if (abs(y_prev - y_real) < Valor):
                    dados = np.append(dados, x_scaler[idx])
                    dadosy = np.append(dadosy, y_real)
                else:
                    dados2 = np.append(dados2, x_scaler[idx])
                    dadosy2 = np.append(dadosy2, y_real)
            # unscale dos dados
            dados, dados2, dadosy, dadosy2 = Tratamento_dados(
                dados, dados2, dadosy, dadosy2, scalerx, scalery)
            # Linear
            reg1 = linear_model.LinearRegression()
            reg2 = linear_model.LinearRegression()
            SSE_final = 0
            media = 5
            for a in range(0, media):
                rand = np.random.randint(0, 100000)
                # Para o cross Validation
                kf = KFold(n_splits=10, shuffle=True, random_state=rand)
                for [(idx_train, idx_teste), (idx_train2, idx_teste2)] in zip(kf.split(dados), kf.split(dados2)):
                    SSE = []
                    reg1.fit(dados[idx_train], dadosy[idx_train])
                    reg2.fit(dados2[idx_train2], dadosy2[idx_train2])
                    y_prever1 = reg1.predict(x)
                    y_prever2 = reg2.predict(x)
                    SSE = np.where((y - y_prever1)**2 > (y-y_prever2)**2,
                                   (y-y_prever2)**2, (y - y_prever1)**2)  # Guardar o melhor SE para cada ponto
                    SSE_final += SSE.sum()
            if (Best_Line[0] > SSE_final):
                Best_Line = [SSE_final, resultados[0],
                             resultados[1], resultados[2]]
                savefinal = dados
                saveyfinal = dadosy
                save2final = dados2
                savey2final = dadosy2
        dados = savefinal
        dadosy = saveyfinal
        dados2 = save2final
        dadosy2 = savey2final
        """ print("\n----------------------------------------------------\nA guardar apenas uma vez os dados!!!!\n")
        np.save("Projeto1_Parte2/Dados/X_train_alpha_regression2.npy", dados)
        np.save("Projeto1_Parte2/Dados/X2_train_alpha_regression2.npy", dados2)
        np.save("Projeto1_Parte2/Dados/y_train_alpha_regression2.npy", dadosy)
        np.save("Projeto1_Parte2/Dados/y2_train_alpha_regression2.npy", dadosy2) """
        SSE_final = Best_Line[0]/(media*kf.get_n_splits())
        Graficox = np.append(Graficox, Valor)
        Graficoy = np.append(Graficoy, SSE_final)
        if (SSE_Min[0] > SSE_final):
            SSE_Min = [SSE_final, Valor]
        print("\nPara o valor de ", round(Valor, 4),
              " o SSE e: ", SSE_final)
        print(
            f"Referencia: Reta 1 entre os pontos {Best_Line[2]}-{Best_Line[3]}\nNumero de pontos 1 = {Best_Line[1]} \t Numero de pontos 2 = {dados2.shape[0]}")
    print("\n\n\nO final de todos foi: ",
          SSE_Min[0], " para o valor de ", round(SSE_Min[1], 4))
    plt.plot(Graficox, Graficoy)
    plt.show()
