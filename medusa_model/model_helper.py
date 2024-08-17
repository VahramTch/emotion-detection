import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import load_img, img_to_array

class FERData:
    def __init__(self, image_size=(48, 48), color_mode='grayscale'):
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
    def __init__(self, train_dir, test_dir, class_labels, train_images, train_labels, test_images, test_labels, image_size=(48, 48), batch_size=64, epochs=50, learning_rate=0.0001):
        """
        Initializes the EmotionRecognitionModel class.

        :param train_dir: Path to the training directory.
        :param test_dir: Path to the testing directory.
        :param class_labels: List of class labels.
        :param image_size: Tuple specifying the size to which each image will be resized.
        :param batch_size: Batch size for training.
        :param epochs: Number of epochs for training.
        :param learning_rate: Learning rate for the optimizer.
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.class_labels = class_labels
        self.num_labels = len(class_labels)
        self.image_size = image_size
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = None
        self.lb = LabelBinarizer()
        self.model_path = os.path.join(os.getcwd(),'medusa_model','h5models','model_optimal.h5')

        # Initialize FERData class for loading images
        self.fer_data = FERData(image_size=self.image_size, color_mode='grayscale')

    def build_model(self):
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

    def train_model(self):
        # Encode labels
        train_labels = self.lb.fit_transform(self.train_labels)
        test_labels = self.lb.transform(self.test_labels)

        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath='checkpoint1.weights.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        )

        # Train the model
        history = self.model.fit(
            self.train_images, train_labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.test_images, test_labels),
            callbacks=[checkpoint_callback]
        )

        # Save the final model
        self.model.save(self.model_path)

        # Plot training & validation accuracy values
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Calculate precision per epoch using validation data
        precisions = []
        for i in range(len(history.history['val_accuracy'])):
            y_pred = self.model.predict(self.test_images)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = np.argmax(test_labels, axis=1)
            precision = precision_score(y_true, y_pred_classes, average='weighted')
            precisions.append(precision)

        plt.subplot(1, 3, 3)
        plt.plot(precisions)
        plt.title('Model Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend(['Validation'], loc='upper left')

        plt.tight_layout()
        plt.show()

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
