import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer


current_dir = os.getcwd()
# Path to the assets directory
assets_dir = os.path.join(current_dir, 'medusa_gui', 'assets')
# Path to the h5models directory
h5models_dir = os.path.join(assets_dir, 'h5models')
# Path to the h5 file inside the 'h5models' folder
h5_model_path = os.path.join(h5models_dir, 'model_optimal.h5')


# Path to the train_model directory
medusa_model = os.path.join(current_dir, 'medusa_model')
# Path to the dataset directory
dataset_dir = os.path.join(medusa_model, 'dataset')
# Path to the train directory
train_dir = os.path.join(dataset_dir, 'train')
# Path to the test directory
test_dir = os.path.join(dataset_dir, 'test')


# Load the pretrain model
model = load_model(h5_model_path)

def load_images_from_directory(directory, image_size, class_labels):
    images = []
    labels = []
    for label in class_labels:
        class_dir = os.path.join(directory, label)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = load_img(image_path, target_size=image_size, color_mode='grayscale')
            image_array = img_to_array(image) / 255.0  # Normalize to [0, 1]
            images.append(image_array)
            labels.append(label)
    return np.array(images), np.array(labels)

# Labels of dataset
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load and preprocess test data
image_size = (48, 48)
test_images, test_labels = load_images_from_directory(test_dir, image_size, class_labels)

# Encode labels
lb = LabelBinarizer()
test_labels_encoded = lb.fit_transform(test_labels)

# Predict the labels for the test set
predictions = model.predict(test_images)
predicted_labels = lb.inverse_transform(predictions)

# Create the confusion matrix
cm = confusion_matrix(test_labels, predicted_labels, labels=class_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
#plt.title('Confusion Matrix')
plt.show()

# Print the classification report
print(classification_report(test_labels, predicted_labels, target_names=class_labels))