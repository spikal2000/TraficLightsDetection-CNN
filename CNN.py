import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Step 1: Prepare the dataset
train_data_dir = 'Traffic_Sign_Dataset\Traffic_Sign_Dataset\Train'
validation_data_dir = 'Traffic_Sign_Dataset\Traffic_Sign_Dataset\Test'
img_width, img_height = 64, 64
batch_size = 32
num_classes = 43

# Step 2: Preprocess the dataset
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Step 3: Design the CNN architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the CNN
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator))

# Step 5: Perform object detection and draw bounding boxes
image_path = 'two.jpg'
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (img_width, img_height))
input_image = np.expand_dims(resized_image, axis=0) / 255.0

predictions = model.predict(input_image)
predicted_class = np.argmax(predictions)

# Draw bounding box
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.gca().add_patch(plt.Rectangle((0, 0), img_width, img_height, linewidth=1, edgecolor='r', facecolor='none'))
plt.text(0, -5, f"Class: {predicted_class}", color='r')
plt.axis('off')
plt.show()
