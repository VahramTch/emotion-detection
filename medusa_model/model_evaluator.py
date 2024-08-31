import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

class ModelEvaluator:
    """
    Initializes the ModelEvaluator class with the model, test data, and labels.

    :param model: Trained Keras model.
    :param test_images: Numpy array of test images.
    :param test_labels: List of true labels for the test images.
    :param class_labels: List of all class labels.
    """
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
        sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=self.class_labels, yticklabels=self.class_labels)
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
        Plots training and validation accuracy and loss in two subplots.
        
        :param history: The history object from Keras model training.
        """
        # Retrieve accuracy and loss values from the history
        acc_train = history.history.get('accuracy', [])
        acc_val = history.history.get('val_accuracy', [])
        loss_train = history.history.get('loss', [])
        loss_val = history.history.get('val_loss', [])
        
        # Determine the range of epochs
        epochs = range(1, len(acc_train) + 1)

        # Create a figure with two subplots (side by side)
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Plot accuracy
        axs[0].plot(epochs, acc_train, label='Training Accuracy', color='blue')
        if acc_val:
            axs[0].plot(epochs, acc_val, label='Validation Accuracy', color='orange')
        axs[0].set_title('Accuracy')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Accuracy')
        axs[0].legend()
        axs[0].grid(True)

        # Plot loss
        axs[1].plot(epochs, loss_train, label='Training Loss', color='blue')
        if loss_val:
            axs[1].plot(epochs, loss_val, label='Validation Loss', color='orange')
        axs[1].set_title('Loss')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')
        axs[1].legend()
        axs[1].grid(True)

        # Adjust layout and show the figure
        plt.tight_layout()
        plt.show()


