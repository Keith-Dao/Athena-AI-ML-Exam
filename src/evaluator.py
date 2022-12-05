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
        print("Generating confusion matrix.")
        display_labels = self.inferencer.get_class_labels()
        ConfusionMatrixDisplay.from_predictions(
            y_true=self.inferencer.get_true_labels(),
            y_pred=self.inferencer.get_predicted_labels(),
            labels=range(len(display_labels)),
            display_labels=display_labels
        )
        plt.show(block=False)
        print("Displaying confusion matrix in a new window.")

    # Calibration Error
    def get_calibration_bins(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Bin confidence, accuracy and count by the confidence in 10 bins.

        Returns:
            The average confidence, accuracy and count bins respectively
        """
        # Constants
        num_bins = 10
        epsilon = 1e-6

        # Bins
        bin_count = torch.zeros(num_bins, dtype=torch.int)
        average_confidence_bins = torch.zeros(num_bins, dtype=torch.float)
        accuracy_bins = torch.zeros(num_bins, dtype=torch.float)

        # Data
        confidences = torch.Tensor(self.inferencer.get_confidences())
        true_positives = (
            torch.Tensor(self.inferencer.get_predicted_labels()) ==
            torch.Tensor(self.inferencer.get_true_labels())
        )

        # Bin based on the confidence
        bins = torch.floor(
            (confidences - epsilon) // (1 / num_bins)
        ).to(torch.int)

        # Calculate bin values
        for bin_id in range(num_bins):
            bin_count[bin_id] = (bins == bin_id).sum().item()
            if bin_count[bin_id] > 0:
                average_confidence_bins[bin_id] = (
                    confidences[bins == bin_id]).sum() / bin_count[bin_id]
                accuracy_bins[bin_id] = (
                    true_positives[bins == bin_id]).sum() / bin_count[bin_id]

        return average_confidence_bins, accuracy_bins, bin_count

    def generate_calibration_error(self) -> None:
        """
        Generate all the values need for calibration error.
        """
        print("Generating calibration error details.")
        average_confidence_bins, accuracy_bins, bin_count = \
            self.get_calibration_bins()

        expected_calibration_error = torch.sum(
            (average_confidence_bins - accuracy_bins).abs() * bin_count / bin_count.sum())
        maximum_calibration_error = torch.max(
            (average_confidence_bins - accuracy_bins).abs())
        print(f"Expected calibration error: {expected_calibration_error}")
        print(f"Maximum calibration error: {maximum_calibration_error}")

    # General
    def print_separator(self) -> None:
        """
        Print a separator in the console.
        """
        print("\n---\n")

    def run(self) -> None:
        """
        Run the evaluation process.
        """
        print("Beginning evaluation.")
        self.print_separator()

        # Inference
        self.inferencer.infer()
        self.print_separator()

        # Confusion matrix
        self.generate_confusion_matrix()
        self.print_separator()

        # Calibration error
        self.generate_calibration_error()
        self.print_separator()

        # Block till all opened windows are closed.
        print("Evaluation completed.")
        print("Waiting for all windows to be closed.")
        plt.show()
        print("All windows closed. Exiting.")
