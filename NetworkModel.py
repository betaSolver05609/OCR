from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from createImage import *
np.random.seed(2019)
#Reading the dataset
df=pd.read_csv('AlphabetsDataset.csv')
df=np.array(df);
#Defining X and Y from Dataset X->Y
Data = df[:,0:784]
LabelSet = df[:,0]
(X_train, X_test, Y_train, Y_test) = train_test_split(Data, LabelSet, test_size=0.50, random_state=seed)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
#Normalizing the dataset
X_train = X_train / 255
X_test = X_test / 255
#using categorical cross entropy
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

num_classes=26 # 26 classes
#Defining NetWorkModel
LearningModel = Sequential()
LearningModeladd(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
LearningModel.add(MaxPooling2D(pool_size=(2, 2)))
LearningModel.add(Dropout(0.2))
LearningModel.add(Flatten())
LearningModel.add(Dense(128, activation='relu'))
LearningModel.add(Dense(num_classes, activation='softmax'))

LearningModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
LearningModel.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)



d=dict({1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G', 8:'H', 9:'I', 10:'J', 11:'K', 12:'L', 13:'M', 14:'N', 15:'O', 16:'P', 17:'Q', 18:'R', 19:'S', 20:'T', 21:'U', 22:'V', 23:'W', 24:'X', 25:'Y', 26:'Z'})


for extract in extracted_image:
    k=cv2.resize(extract, (28,28), interpolation=cv2.INTER_AREA)
    k=k.reshape(1, 28, 28, 1).astype('float32')
    p=LearningModel.predict(k)
    p=p[0]
    p=[i for i,x in enumerate(p) if x>=1]
    p=p[0]+1
    print(d[p], end='')
    print(' ', end='')
    

#X = X.reshape(X.shape[0], 28, 28, 1).astype('float32')
#model.predict(X)


#model.load_weights('C:\\Users\\Dell inspiron\\Desktop\\CurrentWorkingDirectory\\OCR\\my_model_weights.h5')
