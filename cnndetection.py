from __future__ import division, print_function, absolute_import

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tflearn


# Prepare your data
path = 'Traffic_Sign_Dataset\Traffic_Sign_Dataset\Train'
classes = os.listdir(path)
X = []
Y = []
for class_label, class_dir in enumerate(classes):
    class_path = os.path.join(path, class_dir)
    images = os.listdir(class_path)
    for image_name in images:
        image_path = os.path.join(class_path, image_name)
        image = cv2.imread(image_path)  # Load image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        image = cv2.resize(image, (32, 32))  # Resize to match your network architecture
        X.append(image)
        Y.append(class_label)
X = np.array(X) / 255.0  # Normalize pixel values
Y = to_categorical(Y, len(classes))  # Convert labels to one-hot vectors

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Data preprocessing and augmentation
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

# Define network architecture
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, len(classes), activation='softmax')

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Define model
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='sign-classifier.tfl.ckpt')

# Train the model
model.fit(X_train, Y_train, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96, snapshot_epoch=True, run_id='sign-classifier')

# Save the model
model.save("sign-classifier.tfl")
print("Network trained and saved as sign-classifier.tfl!")
