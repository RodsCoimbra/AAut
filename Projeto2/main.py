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

xt = np.load("Dados/Xtrain_Classification1.npy") 
yt = np.load("Dados/ytrain_Classification1.npy").astype(int)
scalerx = StandardScaler().fit(xt)
xScaled = scalerx.transform(xt)

MLP = Sequential()
MLP.add(Convolution2D(32, (3,3), input_shape=(28,28,3), activation='relu'))
MLP.add(MaxPooling2D(pool_size=(2,2)))
MLP.add(Flatten())
MLP.add(Dense(16, activation='relu'))
MLP.add(Dense(16, activation='relu'))
MLP.add(Dense(2, activation='softmax'))	


xScaled = xScaled.reshape(-1,28,28,3)
X_train, X_test, y_train, y_test = train_test_split(xScaled, yt, test_size=0.1)

MLP.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

hist = MLP.fit(x=X_train, y=y_train, epochs=50, validation_data=(X_test, y_test), verbose=2)