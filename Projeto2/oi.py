
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.optimizers import Adam
import cv2
from imblearn.over_sampling import RandomOverSampler, SMOTE
x_meu = np.load("Dados/ytest_Classification1.npy") 
x_joana = np.load("Dados/g43_output_task3.npy")
x_rita = np.load("Dados/ritatio.npy")
x_beta1 = np.load("Dados/yfinal_Classification1_beta.npy")
x_beta_078 = np.load("Dados/yfinal_Classificatio0789.npy")

x_beta_arr = [x_beta1, x_beta_078]
for x_beta in x_beta_arr:
    cm = confusion_matrix(x_beta, x_joana)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    plt.xlabel("Joana")
    plt.ylabel("beta")
    plt.show()

    cm = confusion_matrix(x_beta, x_rita)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    plt.xlabel("Rita")
    plt.ylabel("beta")
    plt.show()

    cm = confusion_matrix(x_meu, x_beta)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    plt.xlabel("Beta")
    plt.ylabel("Eu")
    plt.show()