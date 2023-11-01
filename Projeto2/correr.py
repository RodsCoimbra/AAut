# %%
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import balanced_accuracy_score
import cv2
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import f1_score
# %% [markdown]
# # DADOS

# %%
xt = np.load("Dados/Xtrain_Classification1.npy") 
yt = np.load("Dados/ytrain_Classification1.npy")
xScaled =  (xt).astype('float32')/255.0

# %%
X_train, X_test, y_train, y_test = train_test_split(xScaled, yt, test_size=0.2, shuffle=True, random_state=10, stratify=yt)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=10,stratify=y_train)

# sm = SMOTE(random_state = 2) 
# X_train, y_train = sm.fit_resample(X_train, y_train) 

# %% [markdown]
# # Rodar imagens

# %%
X_testrs = X_test.reshape(-1, 28,28,3)
X_trainrs = X_train.reshape(-1, 28,28,3)
X_validationrs = X_validation.reshape(-1, 28,28,3)
addx = np.array([])
addy = np.array([]) 
for idx, i in enumerate(y_train):
        for l in range(3):
            if l == 0:
                if not np.any(addx):
                    addx = np.expand_dims(cv2.rotate(X_trainrs[idx], cv2.ROTATE_90_COUNTERCLOCKWISE), axis=0)
                    addy = np.append(addy,i)
                else:
                    aux = np.expand_dims(cv2.rotate(X_trainrs[idx], cv2.ROTATE_90_COUNTERCLOCKWISE), axis=0)
                    addx = np.append(addx,aux,axis=0)
                    addy = np.append(addy,i)
                   
            elif l == 1:
                aux = np.expand_dims(cv2.rotate(X_trainrs[idx], cv2.ROTATE_90_CLOCKWISE), axis=0) 
                addx = np.append(addx,aux,axis=0)
                addy = np.append(addy,i)
            
            elif l == 2:
                aux = np.expand_dims(cv2.rotate(X_trainrs[idx],  cv2.ROTATE_180), axis=0) 
                addx = np.append(addx,aux,axis=0)
                addy = np.append(addy,i)

X_trainrs = np.append(X_trainrs,addx,axis=0) 
y_train = np.append(y_train,addy)




# %%

X_trainrs = X_trainrs.reshape(-1, 28*28*3)  
oversample = RandomOverSampler(sampling_strategy='minority')
X_trainrs, y_train = oversample.fit_resample(X_trainrs, y_train)
X_trainrs = X_trainrs.reshape(-1, 28,28,3)
y_train = to_categorical(y_train,2)
y_validation = to_categorical(y_validation,2)
y_test = to_categorical(y_test,2)



# %%
 

# %% [markdown]
# # CNN

for i in [200, 500, 700, 1000, 2000]:
    valor_balanced = 0
    for j in range(0,3):
        MLP = Sequential()
        MLP.add(Convolution2D(16,   kernel_size = 3, activation='relu', input_shape=(28, 28, 3), padding='same'))
        MLP.add(MaxPooling2D((2, 2), strides=2))
        MLP.add(Convolution2D(32, kernel_size = 3, activation='relu', padding='same'))
        MLP.add(MaxPooling2D((2, 2), strides=2))
        MLP.add(Convolution2D(32, kernel_size = 7, activation='relu', padding='same'))
        MLP.add(MaxPooling2D((2, 2), strides=2))
        MLP.add(Flatten())
        MLP.add(Dense(64, activation='relu'))
        MLP.add(Dense(32, activation='relu'))
        MLP.add(Dropout(0.5))
        MLP.add(Dense(32, activation='relu'))
        MLP.add(Dense(2, activation='softmax'))
        MLP.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        Early_callback = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, restore_best_weights=True, min_delta=0.0001)
        hist = MLP.fit(x=X_trainrs, y=y_train, epochs=100, validation_data=(X_validationrs, y_validation), verbose=0, batch_size=i, callbacks=[Early_callback])

        plotx = hist.history['loss']
        plotty = hist.history['val_loss']
        plt.plot(plotx)
        plt.plot(plotty)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'])
        plt.figure()

        y_pred = MLP.predict(X_testrs, verbose=0)
        y_pred = np.argmax(y_pred, axis=1)
        MLP.evaluate(X_testrs, y_test, verbose=1)
        y_comp = np.argmax(y_test, axis=1)
        print("F1 ->", f1_score(y_comp, y_pred))
        valor_balanced += balanced_accuracy_score(y_comp, y_pred)
        print("Balanced ACC ->", balanced_accuracy_score(y_comp, y_pred))
        cm = confusion_matrix(y_comp, y_pred)
        print(cm)

        counter = 0
        for i in range(len(y_comp)):
            if(y_pred[i] != y_comp[i]):
                counter+=1
        counter

        cm = confusion_matrix(y_comp, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
        disp.plot()
    print("\n----------------------------------------------------------------------\n Valor balanced :", valor_balanced/3, "\nPara batch size:", i, "\n\n")
plt.show()



