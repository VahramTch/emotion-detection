import os
from tensorflow.keras.models import load_model
from model_helper import *


# Get the current working directory of the script
current_dir = os.getcwd()

# Path to the 'assets' directory inside the 'medusa_gui' folder
assets_dir = os.path.join(current_dir, 'medusa_gui', 'assets')

# Path to the 'h5models' directory inside the 'assets' folder (used for saving/loading model files)
h5models_dir = os.path.join(assets_dir, 'h5models')

# Path to the 'test' directory inside the 'medusa_model/dataset' folder (used for test data)
test_dir = os.path.join(current_dir, 'medusa_model', 'dataset', 'test')

# Path to the 'train' directory inside the 'medusa_model/dataset' folder (used for training data)
train_dir = os.path.join(current_dir, 'medusa_model', 'dataset', 'train')

# Path to the H5 file ('model_optimal.h5') inside the 'h5models' folder at the current directory level
h5_model_path = os.path.join(os.getcwd(), 'h5models', 'model_optimal.h5')


# Define class labels and image size
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
image_size = (48, 48)

print("Loading the data...")
# Initialize the FERData class
fer_data = FERData(image_size=image_size, color_mode='grayscale')
# Load train data
train_images, train_labels = fer_data.load_images_from_directory(train_dir, class_labels)
# Load test data
test_images, test_labels = fer_data.load_images_from_directory(test_dir, class_labels)
print("Data Loaded !")

# Initialize the EmotionRecognitionModel class
emotion_model = EmotionRecognitionModel(train_dir, test_dir, class_labels, train_images, train_labels, test_images, test_labels, image_size=image_size, batch_size=64, epochs=30, learning_rate=0.0001)
# Build the model
emotion_model.build_model()
# Train the model
history = emotion_model.train_model()

# Load the pre-trained model
model = load_model(h5_model_path)
# Evaluate the model
evaluator = ModelEvaluator(model, test_images, test_labels, class_labels)
evaluator.evaluate()
