import os
import tensorflow as tf
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import load_img, img_to_array


class FERData:
    """
    ============
    FERData Class
    ============

    This class provides functionality for loading and preprocessing images for facial expression recognition.

    Initializing the FERData Class
    -----------------------------

    .. code-block:: python

        def __init__(self, image_size, color_mode='grayscale'):

    Initializes the FERData class with the specified image size and color mode.

    Parameters:
        image_size (tuple): Tuple specifying the size to which each image will be resized (default is (48, 48)).
        color_mode (str): String specifying the color mode, either 'grayscale' or 'rgb' (default is 'grayscale').


    Loading and Preprocessing Images
    -------------------------------

    .. code-block:: python

        def load_images_from_directory(self, directory, class_labels):

    Loads and preprocesses images from the specified directory.

    Parameters:
        directory (str): Path to the directory containing class subdirectories with images.
        class_labels (list): List of class labels (subdirectory names) to load images for.

    Returns:
        tuple: A tuple of (images, labels) where images is a numpy array of image data and labels is a numpy array of corresponding labels.
    """
    def __init__(self, image_size, color_mode='grayscale'):
        """
        Initializes the FERData class with image size and color mode.
        
        :param image_size: Tuple specifying the size to which each image will be resized (default is (48, 48)).
        :param color_mode: String specifying the color mode, either 'grayscale' or 'rgb' (default is 'grayscale').
        """
        self.image_size = image_size
        self.color_mode = color_mode

    def load_images_from_directory(self, directory, class_labels):
        """
        Loads and preprocesses images from the specified directory.

        :param directory: Path to the directory containing class subdirectories with images.
        :param class_labels: List of class labels (subdirectory names) to load images for.
        :return: A tuple of (images, labels) where images is a numpy array of image data and labels is a numpy array of corresponding labels.
        """
        images = []
        labels = []
        for label in class_labels:
            class_dir = os.path.join(directory, label)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image = load_img(image_path, target_size=self.image_size, color_mode=self.color_mode)
                image_array = img_to_array(image) / 255.0  # Normalize to [0, 1]
                images.append(image_array)
                labels.append(label)
        return np.array(images), np.array(labels)


class EmotionRecognitionModel:
    """
    Builds the CNN AlexNet model for emotion recognition.
    """
    def build_alexnet_model(self):
        """
        Builds the CNN AlexNet model for emotion recognition.
        """
        model = Sequential()
        model.add(
            Conv2D(96, kernel_size=(11, 11), input_shape=(self.image_size[0], self.image_size[1], 1), strides=(4, 4),
                   padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(Conv2D(256, kernel_size=(11, 11), strides=(1, 1), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(self.num_labels, activation='softmax'))

        model.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        self.model = model

    def build_cnn_model(self):
        """
        Builds the CNN model for emotion recognition.
        """
        model = Sequential()
        model.add(
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.image_size[0], self.image_size[1], 1)))
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
        model.add(Dense(self.num_labels, activation='softmax'))

        model.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        self.model = model

    def __init__(self, class_labels, train_images, train_labels, valid_images, valid_labels, image_size, batch_size=64,
                 epochs=50, learning_rate=0.0001):
        """
        Initializes the EmotionRecognitionModel class.

        :param class_labels: List of class labels.
        :param image_size: Tuple specifying the size to which each image will be resized.
        :param batch_size: Batch size for training.
        :param epochs: Number of epochs for training.
        :param learning_rate: Learning rate for the optimizer.
        """
        self.class_labels = class_labels
        self.num_labels = len(class_labels)
        self.image_size = image_size
        self.train_images = train_images
        self.train_labels = train_labels
        self.valid_images = valid_images
        self.valid_labels = valid_labels
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = None
        self.lb = LabelBinarizer()
        self.model_path = os.path.join(os.getcwd(), 'medusa_model', 'keras_model',
                                       'model_optimal.keras')  # Updated to .keras format

        # Initialize FERData class for loading images
        self.fer_data = FERData(image_size=self.image_size, color_mode='grayscale')

    def build_lenet5_model(self):
        model = Sequential()

        # First convolutional layer
        model.add(
            Conv2D(6, kernel_size=(5, 5), input_shape=(self.image_size[0], self.image_size[1], 1), activation='relu',
                   padding='same'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(120, activation='relu'))
        model.add(Dense(84, activation='relu'))
        model.add(Dense(self.num_labels, activation='softmax'))

        model.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        self.model = model

    def train_model(self):
        # Encode labels
        train_labels = self.lb.fit_transform(self.train_labels)
        valid_labels = self.lb.transform(self.valid_labels)

        # Stop training when a monitored quantity has stopped improving
        early_stopping_callback = EarlyStopping(monitor='val_loss',
                                                patience=5,
                                                verbose=1,
                                                min_delta=0.01,
                                                restore_best_weights=True)

        # Train the model
        history = self.model.fit(
            self.train_images, train_labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.valid_images, valid_labels), callbacks=[early_stopping_callback]
        )

        # Save the final model in the .keras format
        self.model.save(self.model_path)

        return history
