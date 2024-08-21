import os
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from model_helper import *

# Get the current working directory of the script
current_dir = os.getcwd()

# Path to the 'assets' directory inside the 'medusa_gui' folder
assets_dir = os.path.join(current_dir, 'medusa_gui', 'assets')

# Path to the 'test' directory inside the 'medusa_model/dataset' folder (used for test data)
test_dir = os.path.join(current_dir, 'medusa_model', 'dataset', 'test')

# Path to the 'train' directory inside the 'medusa_model/dataset' folder (used for training data)
train_dir = os.path.join(current_dir, 'medusa_model', 'dataset', 'train')

# Path to the 'keras_model' directory inside the 'medusa_model' folder (used for saving/loading model files)
keras_model_dir = os.path.join(current_dir, 'medusa_model', 'keras_model')

# Path to the keras file ('model_optimal.keras') inside the 'keras_model' folder at the current directory level
keras_model_path = os.path.join(keras_model_dir, 'model_optimal.keras')

# Define class labels
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# Define image size
image_size = (48, 48)

print("Loading the data...")
# Initialize the FERData class
fer_data = FERData(image_size=image_size, color_mode='grayscale')
# Load train data
all_train_images, all_train_labels = fer_data.load_images_from_directory(train_dir, class_labels)
# Load test data
test_images, test_labels = fer_data.load_images_from_directory(test_dir, class_labels)
print("Data Loaded !")

# Split the traing folder into training and validation sets (80% train, 20% validation)
train_images, valid_images, train_labels, valid_labels = train_test_split(
    all_train_images, all_train_labels, test_size=0.2, random_state=42, stratify=all_train_labels
)

# Initialize the EmotionRecognitionModel class
emotion_model = EmotionRecognitionModel(class_labels, train_images, train_labels, valid_images, valid_labels,
                                        image_size=image_size, batch_size=64, epochs=50, learning_rate=0.0001)
# Build the model
emotion_model.build_cnn_model()
# Train the model
history = emotion_model.train_model()

# Load the pre-trained model
model = load_model(keras_model_path)
# Optionally recompile the model to avoid the warning
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])
# Evaluate the model
evaluator = ModelEvaluator(model, test_images, test_labels, class_labels)
evaluator.evaluate()
# Plot the history of the model
evaluator.plot_keras_history(history)
