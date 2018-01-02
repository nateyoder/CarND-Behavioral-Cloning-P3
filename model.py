import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
from keras.utils.visualize_util import plot
from sklearn.utils import shuffle


def generator(data_dir, samples, batch_size=64):
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '{}/IMG/{}'.format(data_dir, batch_sample[0].split('/')[-1])
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # Augment by flipping which effectively doubles the data size
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)


training_data_dir = './clean_data'
samples = []
with open('{}/driving_log.csv'.format(training_data_dir)) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.1)

batch_size = 32
img_shape = (160, 320, 3)

# compile and train the model using the generator function
train_generator = generator(training_data_dir, train_samples, batch_size=batch_size)
validation_generator = generator(training_data_dir, validation_samples, batch_size=batch_size)

model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=img_shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 5, subsample=(1, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

plot(model, to_file='model.png', show_shapes=True)

model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples) * 2,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples) * 2,
                    nb_epoch=10)

model.save('model.h5')
