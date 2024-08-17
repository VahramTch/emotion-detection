import os
from tensorflow.keras.models import load_model
from medusa_model.model_helper import FERData
from medusa_model.model_helper import EmotionRecognitionModel
from medusa_model.model_helper import ModelEvaluator


# Paths and other parameters
current_dir = os.getcwd()
assets_dir = os.path.join(current_dir, 'medusa_gui', 'assets')
h5models_dir = os.path.join(assets_dir, 'h5models')
h5_model_path = os.path.join(h5models_dir, 'model_optimal.h5')
test_dir = os.path.join(current_dir, 'medusa_model', 'dataset', 'test')
train_dir = os.path.join(current_dir, 'medusa_model', 'dataset', 'train')

# Define class labels and image size
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
image_size = (48, 48)

# Initialize the FERData class
fer_data = FERData(image_size=image_size, color_mode='grayscale')
# Load train data
train_images, train_labels = fer_data.load_images_from_directory(train_dir, class_labels)
# Load test data
test_images, test_labels = fer_data.load_images_from_directory(test_dir, class_labels)


# Initialize the EmotionRecognitionModel class
emotion_model = EmotionRecognitionModel(train_dir, test_dir, class_labels, image_size=(48, 48), batch_size=64, epochs=50, learning_rate=0.0001)
# Build the model
emotion_model.build_model()
# Train the model
history = emotion_model.train_model()
print(history)


# Load the pretrained model
model = load_model(h5_model_path)
# Evaluate the model
evaluator = ModelEvaluator(model, test_images, test_labels, class_labels)
evaluator.evaluate()
