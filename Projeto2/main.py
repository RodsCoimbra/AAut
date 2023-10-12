from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


xtrain = np.load("Projeto2/Dados/Xtrain_Classification1.npy") 
ytrain = np.load("Projeto2/Dados/ytrain_Classification1.npy") 

xtrain = xtrain.reshape(-1,28,28,3)

# for i in xtrain:
#  plt.imshow(i)
#  plt.show()

scalerx = StandardScaler().fit(xtrain)
xScaled = scalerx.transform(xtrain)
scalery = StandardScaler().fit(ytrain)
yScaled = scalery.transform(ytrain)
