import csv
import cv2
import numpy as np
import pickle

images = []
measurements = []
cache_dir = './my_data/'
img_dir = './my_data/IMG'
with open('./my_data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        source_path = line[0]
        img_filename = source_path.split('/')[-1]
        img_path = '{}/{}'.format(img_dir, img_filename)
        image = np.array(cv2.imread(img_path))
        angle = float(line[3])

        images.append(image.copy())
        measurements.append(angle)

X_train = np.array(images)
y_train = np.array(measurements)
with open('{}features.pickle'.format(cache_dir), 'wb') as cache:
    pickle.dump({'X': X_train, 'y': y_train}, cache)
