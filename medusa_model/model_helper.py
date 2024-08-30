import os
import tensorflow as tf
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Dense, Dropout, Flatten, concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import load_img, img_to_array


class FERData:
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


    def build_googlenet_model(self):
        input_layer = Input(shape=(self.image_size[0], self.image_size[1], 1))

        conv1 = Conv2D(64, (7,7), strides=(2,2), padding='same', activation='relu')(input_layer)
        maxpool1 = MaxPooling2D((3,3), strides=(2,2), padding='same')(conv1)

        conv2_reduce = Conv2D(64, (1,1), padding='same', activation='relu')(maxpool1)
        conv2 = Conv2D(192, (3,3), padding='same', activation='relu')(conv2_reduce)
        maxpool2 = MaxPooling2D((3,3), strides=(2,2), padding='same')(conv2)

        inception3a = self.inception_module(maxpool2, [64, 96, 128, 16, 32, 32])
        inception3b = self.inception_module(inception3a, [128, 128, 192, 32, 96, 64])
        maxpool3 = MaxPooling2D((3,3), strides=(2,2), padding='same')(inception3b)

        inception4a = self.inception_module(maxpool3, [192, 96, 208, 16, 48, 64])
        inception4b = self.inception_module(inception4a, [160, 112, 224, 24, 64, 64])
        inception4c = self.inception_module(inception4b, [128, 128, 256, 24, 64, 64])
        inception4d = self.inception_module(inception4c, [112, 144, 288, 32, 64, 64])
        inception4e = self.inception_module(inception4d, [256, 160, 320, 32, 128, 128])
        maxpool4 = MaxPooling2D((3,3), strides=(2,2), padding='same')(inception4e)

        inception5a = self.inception_module(maxpool4, [256, 160, 320, 32, 128, 128])
        inception5b = self.inception_module(inception5a, [384, 192, 384, 48, 128, 128])

        # Adjusted pooling size
        avgpool = AveragePooling2D((2,2), strides=(1,1), padding='valid')(inception5b)
        dropout = Dropout(0.4)(avgpool)
        flatten = Flatten()(dropout)
        output_layer = Dense(units=7, activation='softmax')(flatten)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss="categorical_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        self.model = model


    def inception_module(self, x, filters):
        conv1x1 = Conv2D(filters[0], (1,1), padding='same', activation='relu')(x)
        conv3x3_reduce = Conv2D(filters[1], (1,1), padding='same', activation='relu')(x)
        conv3x3 = Conv2D(filters[2], (3,3), padding='same', activation='relu')(conv3x3_reduce)
        conv5x5_reduce = Conv2D(filters[3], (1,1), padding='same', activation='relu')(x)
        conv5x5 = Conv2D(filters[4], (5,5), padding='same', activation='relu')(conv5x5_reduce)
        maxpool = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
        maxpool_conv = Conv2D(filters[5], (1,1), padding='same', activation='relu')(maxpool)
        return concatenate([conv1x1, conv3x3, conv5x5, maxpool_conv], axis=3)
    
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
