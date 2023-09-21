import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x = np.load("Projeto1/X_train_regression1.npy")
y = np.load("Projeto1/y_train_regression1.npy")
x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
Betas = (np.linalg.inv(x.T @ x) @ x.T) @ y
print("Os valores dos Betas são:\n", Betas)
SSE = np.linalg.norm(y-x@Betas)**2
SSE2 = (y - y.mean()).T @ (y - y.mean())
""" SSE3 = (y - x @ Betas).T @ (y - x @ Betas)
SSE4 = ((y-y_avg)**2).sum() """ #Duas maneiras extras de computar o SSE e o SSE2, não sei qual a mais rápida
print()
print("O r^2 é ", 1 - (SSE/SSE2)[0,0])

y_predicted = x @ Betas
print("Os valores preditos são:\n", y_predicted)

""" df = pd.DataFrame()
df['X'] = x[:, 0]
df['Y'] = y
df = df.sort_values(by=['X'])
df.to_csv("Teste.csv", index=False)
df1 = pd.read_csv("Teste.csv")
for i in range(10):
    plt.subplot(5, 2, 1+i)
    plt.scatter(df1['X'], df1['Y'])
plt.show() """
""" a = (y - x @ Betas).T @ (y - x @ Betas)
print(a)  """