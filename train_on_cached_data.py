import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout

cache_dir = './my_data'

with open('{}/features.pickle'.format(cache_dir), 'rb') as cache:
    data_dict = pickle.load(cache)
    X_train = data_dict['X']
    y_train = data_dict['y']

# Augment data
X_train = np.concatenate((X_train, np.fliplr(X_train)), axis=0)
y_train = np.concatenate((y_train, -y_train), axis=0)

del data_dict

img_shape = X_train.shape[1:]

model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=img_shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')
