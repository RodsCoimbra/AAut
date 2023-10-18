import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.metrics import binary_crossentropy
from sklearn.model_selection import train_test_split

xt = np.load("Projeto2/Dados/Xtrain_Classification1.npy") 
yt = np.load("Projeto2/Dados/ytrain_Classification1.npy")
print(yt)
scalerx = StandardScaler().fit(xt)
xScaled = scalerx.transform(xt)
Doencas = ['nevu','melanoma']

MLP = Sequential()
MLP.add(Convolution2D(32, (3,3), input_shape=(28,28,3), activation='relu'))
MLP.add(MaxPooling2D(pool_size=(2,2)))
MLP.add(Convolution2D(32, (3,3), activation='relu'))
MLP.add(MaxPooling2D(pool_size=(2,2)))
MLP.add(Convolution2D(32, (3,3), activation='relu'))
MLP.add(Flatten())
MLP.add(Dense(16, activation='relu'))
MLP.add(Dense(2, activation='softmax'))	


xScaled = xScaled.reshape(-1,28,28,3)
X_train, X_test, y_train, y_test = train_test_split(xScaled, yt, test_size=0.1)
print(y_train)

print(X_test[0:1,0:28,0:28,0:3].shape)

MLP.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

MLP.fit(x=X_train, y=y_train, epochs=25, validation_data=(xScaled, yt))

prediction = MLP.predict(X_test)
print(prediction)

counter0 = 0
counter1 = 0

for i in prediction:
    if i[0]>i[1]:
        counter0+=1
       
    if i[1]>i[0]:
        counter1+=1

counterf0 = 0
counterf1 = 0

for i in y_test:
    if i == 1:
        counterf1+=1
    else:
        counterf0+=1

print(counter0)
print(counter1)
print(counterf0)
print(counterf1)