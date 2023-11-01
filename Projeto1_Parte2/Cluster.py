from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import linear_model

x1 = np.load("Projeto1_Parte2/Dados/X_train_regression2.npy")
x2 = np.load("Projeto1_Parte2/Dados/X_test_regression2.npy")
y = np.load("Projeto1_Parte2/Dados/y_train_regression2.npy")
kmeans = KMeans(n_clusters=2)  # Specify the number of clusters (K)
cluster_labels = kmeans.fit_predict(x1, y)  # X is your data
ln = linear_model.LinearRegression()


print(kmeans.labels_)
print(kmeans.cluster_centers_)
xc1 = []
xc2 = []
yc1 = []
yc2 = []
count = 0
for idx, i in enumerate(kmeans.labels_):
    if i == 0:
        xc1 = np.append(xc1, x1[idx])
        yc1 = np.append(yc1, y[idx])
        count += 1
    if i == 1:
        xc2 = np.append(xc2, x1[idx])
        yc2 = np.append(yc2, y[idx])

xc2 = xc2.reshape(round(xc2.size/4), 4)
yc2 = yc2.reshape(-1, 1)
xc1 = xc2.reshape(round(xc2.size/4), 4)
yc1 = yc2.reshape(-1, 1)
np.save("Projeto1_Parte2/Dados/X_train_beta_regression2.npy", xc1)
np.save("Projeto1_Parte2/Dados/y_train_beta_regression2.npy", yc1)
np.save("Projeto1_Parte2/Dados/X2_train_beta_regression2.npy", xc2)
np.save("Projeto1_Parte2/Dados/y2_train_beta_regression2.npy", yc2)
print("Count = " + str(count))
