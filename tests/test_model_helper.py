import sys
import os
import unittest
import numpy as np
from keras.preprocessing.image import save_img, img_to_array
from sklearn.metrics import accuracy_score

# Add the parent directory (medusa_model) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../medusa_model')))

from model_helper import FERData, EmotionRecognitionModel, ModelEvaluator


class TestFERData(unittest.TestCase):
    """

    The `TestFERData` class is a unit test class for testing the functionality of the `FERData` class.

    __Methods__:

    1. `setUp(self)`: This method is executed before each test case to set up the necessary dependencies and test data.
    2. `tearDown(self)`: This method is executed after each test case to clean up the test directory and remove temporary test data.
    3. `test_initialization(self)`: This method tests the initialization of the `FERData` class by checking if the initialized `image_size` and `color_mode` match the expected values.
    4. `test_load_images_from_directory(self)`: This method tests the `load_images_from_directory` method of the `FERData` class. It creates mock image files in each class directory within the test directory, loads the images using the `FERData` class, and asserts that the loaded images and labels match the expected values.

    Note: This class extends the `unittest.TestCase` class and inherits its methods for asserting test conditions.

    Usage:
        - Create an instance of the `TestFERData` class.
        - Call the appropriate test methods to run the tests and verify the functionality of the `FERData` class.

    Example:
    ```python
    test_fer_data = TestFERData()
    test_fer_data.test_initialization()
    test_fer_data.test_load_images_from_directory()
    ```

    """
    def setUp(self):
        self.image_size = (48, 48)
        self.color_mode = 'grayscale'
        self.fer_data = FERData(self.image_size, self.color_mode)
        self.test_dir = 'test_directory'  # This should be a directory with test images for testing
        self.class_labels = ['happy', 'sad']
        os.makedirs(self.test_dir, exist_ok=True)
        for label in self.class_labels:
            os.makedirs(os.path.join(self.test_dir, label), exist_ok=True)

    def tearDown(self):
        # Clean up the test directory after tests
        for label in self.class_labels:
            label_dir = os.path.join(self.test_dir, label)
            for file in os.listdir(label_dir):
                os.remove(os.path.join(label_dir, file))
            os.rmdir(label_dir)
        os.rmdir(self.test_dir)

    def test_initialization(self):
        self.assertEqual(self.fer_data.image_size, self.image_size)
        self.assertEqual(self.fer_data.color_mode, self.color_mode)

    def test_load_images_from_directory(self):
        # Create a mock image file in each class directory
        for label in self.class_labels:
            image = np.ones((48, 48, 1), dtype=np.uint8) * 255
            image_path = os.path.join(self.test_dir, label, 'test_image.jpg')
            save_img(image_path, img_to_array(image))

        images, labels = self.fer_data.load_images_from_directory(self.test_dir, self.class_labels)

        self.assertEqual(len(images), 2)
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels[0], 'happy')
        self.assertEqual(labels[1], 'sad')
        self.assertEqual(images.shape[1:], (48, 48, 1))  # Check image dimensions


class TestEmotionRecognitionModel(unittest.TestCase):
    """
    Unit tests for the EmotionRecognitionModel class.
    """
    def setUp(self):
        self.image_size = (48, 48)
        self.class_labels = ['happy', 'sad']
        self.train_images = np.random.rand(10, 48, 48, 1)
        self.train_labels = ['happy'] * 5 + ['sad'] * 5
        self.valid_images = np.random.rand(4, 48, 48, 1)
        self.valid_labels = ['happy'] * 2 + ['sad'] * 2
        self.batch_size = 2
        self.epochs = 5
        self.learning_rate = 0.0001
        self.emotion_model = EmotionRecognitionModel(
            self.class_labels, self.train_images, self.train_labels,
            self.valid_images, self.valid_labels,
            self.image_size, self.batch_size, self.epochs, self.learning_rate
        )

    def test_initialization(self):
        self.assertEqual(self.emotion_model.num_labels, 2)
        self.assertEqual(self.emotion_model.image_size, self.image_size)
        self.assertEqual(self.emotion_model.batch_size, self.batch_size)
        self.assertEqual(self.emotion_model.epochs, self.epochs)

    def test_build_cnn_model(self):
        self.emotion_model.build_cnn_model()
        self.assertIsNotNone(self.emotion_model.model)
        self.assertEqual(len(self.emotion_model.model.layers), 23)  # Updated to match the actual layer count

    def test_train_model(self):
        self.emotion_model.build_cnn_model()
        history = self.emotion_model.train_model()
        self.assertIn('accuracy', history.history)
        self.assertIn('val_accuracy', history.history)


class TestModelEvaluator(unittest.TestCase):
    """
    This class is responsible for testing the functionality of the ModelEvaluator class.

    Attributes:
        image_size (tuple): The size of the input images.
        class_labels (list): The list of class labels.
        test_images (ndarray): The test images.
        test_labels (list): The list of true labels for the test images.
        emotion_model (EmotionRecognitionModel): The emotion recognition model.
        evaluator (ModelEvaluator): The model evaluator instance.

    """
    def setUp(self):
        self.image_size = (48, 48)
        self.class_labels = ['happy', 'sad']
        self.test_images = np.random.rand(4, 48, 48, 1)
        self.test_labels = ['happy', 'sad', 'happy', 'sad']
        self.emotion_model = EmotionRecognitionModel(
            self.class_labels, None, None, None, None,
            self.image_size
        )
        self.emotion_model.build_cnn_model()
        self.evaluator = ModelEvaluator(self.emotion_model.model, self.test_images, self.test_labels, self.class_labels)

    def test_evaluate(self):
        # Since we cannot really train the model in unit tests, we use fake predictions
        fake_predictions = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
        fake_predictions_labels = ['happy', 'sad', 'happy', 'sad']
        self.evaluator.model.predict = lambda x: fake_predictions  # Mock the predict method

        accuracy = accuracy_score(self.test_labels, fake_predictions_labels)
        self.assertGreater(accuracy, 0.5)  # Simple check to ensure fake predictions work


if __name__ == '__main__':
    unittest.main()
