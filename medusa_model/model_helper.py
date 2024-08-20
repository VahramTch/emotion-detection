import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Multiply, GlobalAveragePooling2D, Reshape, Dense, Dropout, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
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
    def __init__(self, class_labels, train_images, train_labels, valid_images, valid_labels, image_size, batch_size=64, epochs=50, learning_rate=0.0001):
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
        self.valid_images = valid_images,
        self.valid_labels = valid_labels
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = None
        self.lb = LabelBinarizer()
        self.model_path = os.path.join(os.getcwd(),'medusa_model','h5models','model_optimal.h5')

        # Initialize FERData class for loading images
        self.fer_data = FERData(image_size=self.image_size, color_mode='grayscale')
    
    def build_cnn_model(self):
        """
        Builds the CNN model for emotion recognition.
        """
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.image_size[0], self.image_size[1], 1)))
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

        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        self.model = model

    def build_alexnet_model(self):
        """
        Builds the CNN AlexNet model for emotion recognition.
        """
        model = Sequential()
        model.add(Conv2D(96, kernel_size=(11,11), input_shape=(self.image_size[0], self.image_size[1], 1), strides=(4,4), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))                 
        model.add(Conv2D(256, kernel_size=(11,11), strides=(1,1), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Conv2D(384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add(Conv2D(384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(self.num_labels, activation='softmax'))

        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        self.model = model

    def build_lenet5_model(self):

        model = Sequential()

        # First convolutional layer
        model.add(Conv2D(6, kernel_size=(5, 5), input_shape=(self.image_size[0], self.image_size[1], 1), activation='relu', padding='same'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(120, activation='relu'))
        model.add(Dense(84, activation='relu'))
        model.add(Dense(self.num_labels, activation='softmax'))

        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        self.model = model


    def train_model(self):
        # Encode labels
        train_labels = self.lb.fit_transform(self.train_labels)
        valid_labels = self.lb.transform(self.valid_labels)

        # Stop training when a monitored quantity has stopped improving
        early_stopping_callback = EarlyStopping(monitor='val_loss',
                  patience=3,
                  verbose=1,
                  min_delta=0.01,
                  restore_best_weights=True)

        # Train the model
        history = self.model.fit(
            self.train_images, train_labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.valid_images, valid_labels),callbacks=[early_stopping_callback]
        )

        # Save the final model
        self.model.save(self.model_path)

        return history

class ModelEvaluator:
    def __init__(self, model, test_images, test_labels, class_labels):
        """
        Initializes the ModelEvaluator class with the model, test data, and labels.
        
        :param model: Trained Keras model.
        :param test_images: Numpy array of test images.
        :param test_labels: List of true labels for the test images.
        :param class_labels: List of all class labels.
        """
        self.model = model
        self.test_images = test_images
        self.test_labels = test_labels
        self.class_labels = class_labels
        self.lb = LabelBinarizer()
        self.test_labels_encoded = self.lb.fit_transform(test_labels)

    def evaluate(self):
        """
        Evaluates the model on the test data and prints the confusion matrix, classification report, and metrics.
        """
        # Predict the labels for the test set
        predictions = self.model.predict(self.test_images)
        predicted_labels = self.lb.inverse_transform(predictions)
        
        # Create the confusion matrix
        cm = confusion_matrix(self.test_labels, predicted_labels, labels=self.class_labels)
        
        # Plot the confusion matrix
        self.plot_confusion_matrix(cm)
        
        # Print the classification report
        print("Classification Report:\n")
        print(classification_report(self.test_labels, predicted_labels, target_names=self.class_labels))
        
        # Calculate and print metrics
        self.print_metrics(self.test_labels, predicted_labels)

    def plot_confusion_matrix(self, cm):
        """
        Plots the confusion matrix.
        
        :param cm: Confusion matrix to be plotted.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_labels, yticklabels=self.class_labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def print_metrics(self, true_labels, predicted_labels):
        """
        Calculates and prints accuracy, F1-score, precision, and recall.
        
        :param true_labels: List of true labels.
        :param predicted_labels: List of predicted labels by the model.
        """
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        accuracy = accuracy_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels, average='weighted')

        print(f"F1-score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

    def plot_keras_history(self, history):
        """
        :param history: 
        :return: 
        """
        # the history object gives the metrics keys. 
        # we will store the metrics keys that are from the training sesion.
        metrics_names = [key for key in history.history.keys() if not key.startswith('val_')]

        for i, metric in enumerate(metrics_names):
            
            # getting the training values
            metric_train_values = history.history.get(metric, [])
            
            # getting the validation values
            metric_val_values = history.history.get("val_{}".format(metric), [])

            # As loss always exists as a metric we use it to find the 
            epochs = range(1, len(metric_train_values) + 1)
            
            # leaving extra spaces to allign with the validation text
            training_text = "   Training {}: {:.5f}".format(metric, metric_train_values[-1])

            # metric
            plt.figure(i, figsize=(12, 6))
            plt.plot(epochs, metric_train_values, 'b', label=training_text)
            
            # if we validation metric exists, then plot that as well
            if metric_val_values:
                validation_text = "Validation {}: {:.5f}".format(metric, metric_val_values[-1])
                plt.plot(epochs, metric_val_values, 'g', label=validation_text)
            
            # add title, xlabel, ylabe, and legend
            plt.title('Model Metric: {}'.format(metric))
            plt.xlabel('Epochs')
            plt.ylabel(metric.title())
            plt.legend()

        plt.show()