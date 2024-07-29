import os
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

class ModelHelper:

    def get_model(self):
        # Define the model architecture
        model = Sequential()

        # Add a convolutional layer with 32 filters, 3x3 kernel size, and relu activation function
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        # Add a batch normalization layer
        model.add(BatchNormalization())
        # Add a second convolutional layer with 64 filters, 3x3 kernel size, and relu activation function
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        # Add a second batch normalization layer
        model.add(BatchNormalization())
        # Add a max pooling layer with 2x2 pool size
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Add a dropout layer with 0.25 dropout rate
        model.add(Dropout(0.25))

        # Add a third convolutional layer with 128 filters, 3x3 kernel size, and relu activation function
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        # Add a third batch normalization layer
        model.add(BatchNormalization())
        # Add a fourth convolutional layer with 128 filters, 3x3 kernel size, and relu activation function
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        # Add a fourth batch normalization layer
        model.add(BatchNormalization())
        # Add a max pooling layer with 2x2 pool size
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Add a dropout layer with 0.25 dropout rate
        model.add(Dropout(0.25))

        # Add a fifth convolutional layer with 256 filters, 3x3 kernel size, and relu activation function
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        # Add a fifth batch normalization layer
        model.add(BatchNormalization())
        # Add a sixth convolutional layer with 256 filters, 3x3 kernel size, and relu activation function
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        # Add a sixth batch normalization layer
        model.add(BatchNormalization())
        # Add a max pooling layer with 2x2 pool size
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Add a dropout layer with 0.25 dropout rate
        model.add(Dropout(0.25))

        # Flatten the output of the convolutional layers
        model.add(Flatten())
        # Add a dense layer with 256 neurons and relu activation function
        model.add(Dense(256, activation='relu'))
        # Add a seventh batch normalization layer
        model.add(BatchNormalization())
        # Add a dropout layer with 0.5 dropout rate
        model.add(Dropout(0.5))
        # Add a dense layer with 7 neurons (one for each class) and softmax activation function
        model.add(Dense(7, activation='softmax'))
                
        return model
