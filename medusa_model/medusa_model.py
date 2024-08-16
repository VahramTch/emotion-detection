import tensorflow as tf
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelBinarizer

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
#image_size = 224
num_features = 64
batch_size = 64
epochs = 50
width, height = 48, 48
image_size = (48, 48)

def load_images_from_directory(directory, image_size, class_labels):
    images = []
    labels = []
    for label in class_labels:
        class_dir = os.path.join(directory, label)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = load_img(image_path, target_size=image_size, color_mode='grayscale')
            image_array = img_to_array(image) / 255.0  # Normalize to [0, 1]
            images.append(image_array)
            labels.append(label)
    return np.array(images), np.array(labels)

# Define class labels
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load and preprocess training and test data
train_images, train_labels = load_images_from_directory(train_dir, image_size, class_labels)
test_images, test_labels = load_images_from_directory(test_dir, image_size, class_labels)

# Encode labels
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.transform(test_labels)


# Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

checkpoint_callback = ModelCheckpoint(
    filepath='checkpoint1.weights.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1
)

history = model.fit(
    train_images, train_labels,
    batch_size=64,
    epochs=50,
    validation_data=(test_images, test_labels),
    callbacks=[checkpoint_callback]
)

model.save('model_optimal.h5')
