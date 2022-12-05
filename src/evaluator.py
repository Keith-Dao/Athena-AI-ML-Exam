"""
    Model and dataset evaluator module.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch

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

    # Confusion Matrix
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

    # Calibration Error
    def generate_calibration_error(self) -> None:
        """
        Generate all the values need for calibration error.
        """
        num_bins = 10
        epsilon = 1e-6
        bin_count = torch.zeros(num_bins, dtype=torch.int)
        confidence_bins = torch.zeros(num_bins, dtype=torch.float)
        accuracy_bins = torch.zeros(num_bins, dtype=torch.float)

        confidences = torch.Tensor(self.inferencer.get_confidences())
        true_positives = (
            torch.Tensor(self.inferencer.get_predicted_labels()) ==
            torch.Tensor(self.inferencer.get_true_labels())
        )

        bins = torch.floor(
            (confidences - epsilon) // (1 / num_bins)
        ).to(torch.int)

        for bin_id in range(num_bins):
            bin_count[bin_id] = (bins == bin_id).sum().item()
            if bin_count[bin_id] > 0:
                confidence_bins[bin_id] = (
                    confidences[bins == bin_id]).sum() / bin_count[bin_id]
                accuracy_bins[bin_id] = (
                    true_positives[bins == bin_id]).sum() / bin_count[bin_id]

    # General
    def run(self) -> None:
        """
        Run the evaluation process.
        """
        print("Beginning evaluation.")

        # Inference
        self.inferencer.infer()

        # Confusion matrix
        self.generate_confusion_matrix()

        # Calibration error
        self.generate_calibration_error()

        # Block till all opened windows are closed.
        print("Evaluation completed.")
        print("Waiting for all windows to be closed.")
        plt.show()
        print("All windows closed. Exiting.")
