"""
    Model and dataset evaluator module.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from src.inferencer import Inferencer


class Evaluator:
    """
    Evaluator for a given dataset and model.
    """

    def __init__(self, dataset_path: str, model_size: str) -> None:
        self.inferencer = Inferencer(
            dataset_path=dataset_path,
            model_size=model_size,
        )

    def generate_confusion_matrix(self) -> None:
        """
        Generate the confusion matrix.
        """
        display_labels = self.inferencer.get_class_labels()
        ConfusionMatrixDisplay.from_predictions(
            y_true=self.inferencer.get_true_labels(),
            y_pred=self.inferencer.get_predicted_labels(),
            labels=range(len(display_labels)),
            display_labels=display_labels
        )
        plt.show(block=False)

    def run(self) -> None:
        """
        Run the evaluation process.
        """
        print("Beginning evaluation.")

        # Inference
        self.inferencer.infer()

        # Confusion matrix
        self.generate_confusion_matrix()

        # Block till all opened windows are closed.
        print("Evaluation completed.")
        print("Waiting for all windows to be closed.")
        plt.show()
        print("All windows closed. Exiting.")
