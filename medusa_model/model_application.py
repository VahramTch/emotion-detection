import os
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from model_helper import *
from model_evaluator import ModelEvaluator
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description="Train a CNN model.")
# Add an argument to specify the model type
parser.add_argument("--model", type=str, choices=["cnn", "googlenet", "lenet5"], default="cnn",
					help="Select the model to train: 'cnn', 'googlenet', or 'lenet5'.")
# Parse command-line arguments
args = parser.parse_args()

# Get the current working directory of the script
current_dir = os.getcwd()

# Path to the 'test' directory inside the 'medusa_model/dataset' folder (used for test data)
test_dir = os.path.join(current_dir, 'medusa_model', 'dataset', 'test')

# Path to the 'train' directory inside the 'medusa_model/dataset' folder (used for training data)
train_dir = os.path.join(current_dir, 'medusa_model', 'dataset', 'train')

# Path to the 'keras_model' directory inside the 'medusa_model' folder (used for saving/loading model files)
keras_model_dir = os.path.join(current_dir, 'medusa_model', 'keras_model')

# Path to the keras file ('model_optimal.keras') inside the 'keras_model' folder at the current directory level
keras_model_path = os.path.join(keras_model_dir, 'model_optimal.keras')

# Path to the keras file ('model_optimal.keras') inside the 'keras_model' folder at the current directory level
train_aug_dir = os.path.join(current_dir, 'medusa_model', 'dataset', 'train_aug')

# Path to the keras file ('model_optimal.keras') inside the 'keras_model' folder at the current directory level
test_compl_dir = os.path.join(current_dir, 'medusa_model', 'dataset', 'test_compl')

# Define class labels
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# Define image size
image_size = (48, 48)


# Initialize the FERData class
fer_data = FERData(image_size=image_size, color_mode='grayscale')
# Load train data
all_train_images, all_train_labels = fer_data.load_images_from_directory(train_dir, class_labels)
# Load test data
test_images, test_labels = fer_data.load_images_from_directory(test_dir, class_labels)

# Plot the class distribution before the data augmentation process.
fer_data.plot_class_distribution(all_train_labels)

fer_data.generate_augmented_images(train_dir, train_aug_dir, class_labels)
# Moves the not augmeneted data from the training directory to the new directory. Does not delete the diles from the training directory.
fer_data.move_files(train_dir, train_aug_dir, class_labels)
all_train_aug_images, all_train_aug_labels = fer_data.load_images_from_directory(train_aug_dir, class_labels)

# Plot the class distribution after the data augmentation process.
fer_data.plot_class_distribution(all_train_aug_labels)

# Split the traing folder into training and validation sets (80% train, 20% validation)
train_images, valid_images, train_labels, valid_labels = train_test_split(
    all_train_aug_images, all_train_aug_labels, test_size=0.2, random_state=42, stratify=all_train_aug_labels)


# Initialize the EmotionRecognitionModel class
emotion_model = EmotionRecognitionModel(class_labels, train_images, train_labels, valid_images, valid_labels,
                                        image_size=image_size, batch_size=64, epochs=50, learning_rate=0.0001)

# Please select the model you want to train here. Custom CNN is the default model.
if args.model == "cnn":
	model = emotion_model.build_cnn_model()
elif args.model == "googlenet":
	model = emotion_model.build_googlenet_model()
elif args.model == "lenet5":
	model = emotion_model.build_lenet5_model()
else:
	raise ValueError("Invalid model choice!")

# Train the model
history = emotion_model.train_model()

# Load the pre-trained model
model = load_model(keras_model_path)


# Evaluate the model
evaluator = ModelEvaluator(model, test_images, test_labels, class_labels)
evaluator.evaluate()
# Plot the history of the model
evaluator.plot_keras_history(history)