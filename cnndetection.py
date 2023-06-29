from __future__ import division, print_function, absolute_import

import os
import glob
import numpy as np
from PIL import Image
import tflearn
from tflearn.data_utils import to_categorical, shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

def load_data(data_dir, num_classes):
    X = []
    Y = []
    for class_id in range(num_classes):
        for filename in glob.glob(os.path.join(data_dir, str(class_id), '*.png')):
            print(f"Loading {filename}")  # Add this line to print out each file being loaded
            im = Image.open(filename)
            im = im.resize((32, 32))
            im = np.array(im)
            X.append(im)
            Y.append(class_id)
    return np.array(X, dtype='float32'), np.array(Y, dtype='int')  # Use float32 for X and int for Y

def load_test_data(data_dir):
    X = []
    filenames = []

    for filename in glob.glob(os.path.join(data_dir, '*.png')):
        print(f"Loading {filename}")  # Add this line to print out each file being loaded
        im = Image.open(filename)
        im = im.resize((32, 32))
        im = np.array(im)
        X.append(im)
        filenames.append(filename)
    return np.array(X, dtype='float32'), filenames

num_classes = 43  # Adjust this to match the number of road sign types in your dataset
data_dir = '../Traffic_Sign_Dataset/Train'
test_dir = '../Traffic_Sign_Dataset/Test/img'

# Load the data
X_test, test_filenames = load_test_data(test_dir)

# Check if the validation data is loaded correctly
if len(X_test) == 0:
    raise Exception("No validation data found. Please check the path to the validation dataset.")
X, Y = load_data(data_dir, num_classes)
# Shuffle the data
X, Y = shuffle(X, Y)

# Convert labels to one-hot vectors
Y = to_categorical(Y, num_classes)
# Y_test = to_categorical(Y, num_classes)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, num_classes, activation='softmax')  # Changed 10 to num_classes
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y), show_metric=True, batch_size=96, run_id='cifar10_cnn')

# Save the model
model.save("road_signs_model.tfl")
