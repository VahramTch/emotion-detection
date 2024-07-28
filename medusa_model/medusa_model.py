import tensorflow as tf
import cv2
import os
import numpy as np
import random
from model_helper import ModelHelper
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

# Get the directory of the current script
current_dir = os.getcwd()
# Path to the medusa_model directory
medusa_model_dir = os.path.join(current_dir, 'medusa_model')
# Path to the dataset directory
dataset_dir = os.path.join(medusa_model_dir, 'dataset')
# Path to the train directory
train_dir = os.path.join(dataset_dir, 'train')
# Path to the test directory
test_dir = os.path.join(dataset_dir, 'test')

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
num_labels = len(classes)
image_size = 224
num_features = 64
batch_size = 64
epochs = 5
width, height = 48, 48

# Initialize the ModelHelper class
medusa_helper = ModelHelper()

# Call the method with the appropriate arguments
training_dataset = medusa_helper.get_array_from_image_dataset(train_dir, classes, image_size)

print(len(training_dataset))

random.shuffle(training_dataset)

x = []
y = []
for features, label in training_dataset:
    x.append(features)
    y.append(label)

x_train = np.array(x).reshape(-1, image_size, image_size, 3)  # Converting it to 4 dimensions

# Create a dictionary to map class names to integers
class_mapping = {cls: idx for idx, cls in enumerate(classes)}

# Convert labels from strings to integers
y_train_int = np.array([class_mapping[label] for label in y])

# Convert integer labels to one-hot encoding
y_train_one_hot = tf.keras.utils.to_categorical(y_train_int, num_classes=num_labels)

# Normalize the data
x_train = x_train / 255.0

# Designing the CNN
model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(image_size, image_size, 3), data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

# model.summary()

# Compiling the model with Adam optimizer and categorical crossentropy loss
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

# Training the model
model.fit(x_train, y_train_one_hot,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          shuffle=True)

# Uncomment to save the model

# Saving the model to be used later
trained_model = model.to_json()
with open("trained_model.json", "w") as json_file:
    json_file.write(trained_model)
model.save_weights("trained_model.h5")
print("Saved model to disk")

