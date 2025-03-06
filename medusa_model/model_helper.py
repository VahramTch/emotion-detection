import os
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import shutil
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Dense, Dropout, Flatten, concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


class FERData:
    """
    FERData represents a class that loads and processes images for Facial Emotion Recognition.

    Args:
        image_size (tuple): The target size of the images (height, width).
        color_mode (str, optional): The color mode of the images. Defaults to 'grayscale'.

    Attributes:
        image_size (tuple): The target size of the images (height, width).
        color_mode (str): The color mode of the images.

    Methods:
        load_images_from_directory(directory, class_labels): Loads the images from a directory and returns the images and labels.

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
        print("Loading data...")
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
        print("Data Loaded !")
        return np.array(images), np.array(labels)
    
    def plot_class_distribution(self, labels):
        """
        Plots a visually enhanced distribution of the dataset for each class.

        :param labels: A numpy array of class labels corresponding to the dataset images.
        """
        unique_classes, counts = np.unique(labels, return_counts=True)
        
        # Set Seaborn style
        sns.set_style("whitegrid")
        
        plt.figure(figsize=(12, 6))
        colors = sns.color_palette("pastel", len(unique_classes))  # Soft color scheme
        bars = plt.bar(unique_classes, counts, color=colors, edgecolor="black", linewidth=1.2)
        
        # Add text labels on top of the bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, 
                    str(count), ha="center", fontsize=12, fontweight="bold", color="black")

        # Customize plot aesthetics
        plt.xlabel("Class", fontsize=14, fontweight="bold")
        plt.ylabel("Number of Images", fontsize=14, fontweight="bold")
        plt.title("Dataset Class Distribution", fontsize=16, fontweight="bold")
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

    def generate_augmented_images(self, source_directory, target_directory, class_labels, target_count=7000):
        """
        This function generates augmented images for each class in the source directory and saves them to the target directory.

        Parameters:
        - source_directory (str): The path to the directory containing subdirectories for each class with the source images.
        - target_directory (str): The path where the augmented images will be saved, organized in subdirectories by class.
        - class_labels (list): A list of class labels (subdirectory names in the source directory) for which augmentation will be applied.
        - target_count (int): The desired number of images per class in the target directory. By default, it is set to 7000.

        The function uses an ImageDataGenerator to apply various transformations (rotation, shift, shear, zoom, and flipping) to the source images,
        and saves the generated augmented images to the target directory. It ensures each class has at least the target_count of images.
        """
        
        print("Generating images...")
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        for label in class_labels:
            class_dir = os.path.join(source_directory, label)
            save_dir = os.path.join(target_directory, label)
            os.makedirs(save_dir, exist_ok=True)

            images = []
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image = load_img(image_path, target_size=self.image_size, color_mode=self.color_mode)
                image_array = img_to_array(image) / 255.0
                images.append(image_array)

            images = np.array(images)
            current_count = len(images)
            needed_images = target_count - current_count

            if needed_images > 0:
                i = 0
                for batch in datagen.flow(images, batch_size=1, save_to_dir=save_dir, save_prefix='aug', save_format='png'):
                    i += 1
                    if i >= needed_images:
                        break

        print("Data augmentation completed.")

    def move_files(self, source_folder, target_folder, class_labels):
        """
        Copies files from subfolders in source_folder to corresponding subfolders in target_folder.

        :param source_folder: Path to the source directory.
        :param target_folder: Path to the target directory.
        :param class_labels: List of subfolder names (class labels).
        """
        print("Copying files..")
        for label in class_labels:
            source_path = os.path.join(source_folder, label)
            target_path = os.path.join(target_folder, label)

            if not os.path.exists(target_path):
                os.makedirs(target_path)  # Create subfolder if it doesn't exist

            for file_name in os.listdir(source_path):
                file_source = os.path.join(source_path, file_name)
                file_target = os.path.join(target_path, file_name)

                if os.path.isfile(file_source):  # Ensure it's a file
                    shutil.copy2(file_source, file_target)  # Copy file with metadata

        print("All files copied successfully!")

class EmotionRecognitionModel:
    """
    Class representing a model for emotion recognition.

    Args:
        class_labels (list): List of class labels.
        train_images (ndarray): Training images.
        train_labels (list): Training labels.
        valid_images (ndarray): Validation images.
        valid_labels (list): Validation labels.
        image_size (tuple): Size of the input images.
        batch_size (int, optional): Batch size for training. Defaults to 64.
        epochs (int, optional): Number of epochs for training. Defaults to 50.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 0.0001.

    Attributes:
        class_labels (list): List of class labels.
        image_size (tuple): Size of the input images.
        train_images (ndarray): Training images.
        train_labels (list): Training labels.
        valid_images (ndarray): Validation images.
        valid_labels (list): Validation labels.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for optimizer.
        model (Sequential): Emotional recognition model.
        lb (LabelBinarizer): Label binarizer.
        model_path (str): Path to save the trained model.
        fer_data (FERData): FERData instance for data preprocessing.

    Methods:
        build_googlenet_model(): Build the AlexNet model.
        build_cnn_model(): Build the CNN model.
        build_lenet5_model(): Build the LeNet-5 model.
        train_model(): Train the model.

    """

    def __init__(self, class_labels, train_images, train_labels, valid_images, valid_labels, image_size, batch_size=64,
                 epochs=50, learning_rate=0.0001):
        """
        Initializes the EmotionRecognitionModel class.
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
        """
        Builds the GoogLeNet (Inception) model for emotion recognition.
        """
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
        """
        Defines the inception module used in GoogLeNet.

        :param x: Input tensor.
        :param filters: List of filter sizes for the different convolution layers in the module.
        :return: Concatenated output tensor from the inception module.
        """
        conv1 = Conv2D(filters[0], (1,1), padding='same', activation='relu')(x)
        conv3_reduce = Conv2D(filters[1], (1,1), padding='same', activation='relu')(x)
        conv3 = Conv2D(filters[2], (3,3), padding='same', activation='relu')(conv3_reduce)
        conv5_reduce = Conv2D(filters[3], (1,1), padding='same', activation='relu')(x)
        conv5 = Conv2D(filters[4], (5,5), padding='same', activation='relu')(conv5_reduce)
        maxpool = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
        maxpool_conv = Conv2D(filters[5], (1,1), padding='same', activation='relu')(maxpool)
        return concatenate([conv1, conv3, conv5, maxpool_conv], axis=3)
    
    def build_lenet5_model(self):
        """
        Builds the LeNet-5 model for emotion recognition.
        """
        model = Sequential()
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
        Builds the custom CNN model for emotion recognition.
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
        """
        Trains the emotion recognition model and saves the best performing model.

        :return: Training history object that contains training metrics.
        """
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
