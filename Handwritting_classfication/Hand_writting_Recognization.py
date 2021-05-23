import pandas as pd
import numpy as np
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dense,Dropout,Flatten
from keras.optimizers import Adam,SGD
from keras.utils import to_categorical
from keras.models import Sequential,load_model
import keras
from keras.metrics import mean_squared_error
import matplotlib.pyplot as plt

(X_train,y_train),(X_test,y_test)=mnist.load_data()
print(X_train.shape)
print(X_test.shape)

# Reshape the data
X_train=X_train.reshape(X_train.shape[0],28,28,1)
X_test=X_test.reshape(X_test.shape[0],28,28,1)
input_shape=(X_train.shape[1:])


# to_categorical == ONEHOTENCODER
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

plt.imshow(X_train[0])
plt.show()

# define the model
batch_size = 128
num_classes = 10
epochs = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=128,epochs=10,verbose=1,validation_data=(X_test,y_test))

# evalutation
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# model saving
model.save("model.h5")