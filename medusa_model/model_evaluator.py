import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

class ModelEvaluator:
    """
    ModelEvaluator class

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

            # leaving extra spaces to align with the validation text
            training_text = "   Training {}: {:.5f}".format(metric, metric_train_values[-1])

            # metric
            plt.figure(i, figsize=(12, 6))
            plt.plot(epochs, metric_train_values, 'b', label=training_text)

            # if we validate metric exists, then plot that as well
            if metric_val_values:
                validation_text = "Validation {}: {:.5f}".format(metric, metric_val_values[-1])
                plt.plot(epochs, metric_val_values, 'g', label=validation_text)

            # add title, xlabel, ylabe, and legend
            plt.title('Model Metric: {}'.format(metric))
            plt.xlabel('Epochs')
            plt.ylabel(metric.title())
            plt.legend()

        plt.show()
