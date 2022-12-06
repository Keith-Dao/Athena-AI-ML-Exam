"""
    Model and dataset evaluator module.
"""
import os

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

        # Create the results folder, if it does not exist
        self.results_folder = "results"
        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)

    # Confusion Matrix
    def display_confusion_matrix(self) -> None:
        """
        Display the confusion matrix.
        """
        print("Generating confusion matrix.")

        display_labels = self.inferencer.get_class_labels()
        figure, axis = plt.subplots()
        figure.canvas.manager.set_window_title('Confusion matrix')

        ConfusionMatrixDisplay.from_predictions(
            y_true=self.inferencer.get_true_labels(),
            y_pred=self.inferencer.get_predicted_labels(),
            labels=range(len(display_labels)),
            display_labels=display_labels,
            ax=axis
        )
        figure.show()

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
                    (confidences[bins == bin_id]).sum() / bin_count[bin_id]
                )
                accuracy_bins[bin_id] = (
                    (true_positives[bins == bin_id]).sum() / bin_count[bin_id]
                )

        return average_confidence_bins, accuracy_bins, bin_count

    def get_calibration_graph(self, accuracy_bins: torch.Tensor) -> plt.Figure:
        """
        Get the calibration graph.

        Args:
            accuracy_bins (tensor): The accuracy of each confidence bin
        Returns:
            The calibration graph figure
        """

        num_bins = len(accuracy_bins)
        confidence_bins = [
            confidence / 100
            for confidence in range(0, 100, 100 // num_bins)
        ]

        # Plot
        figure, axis = plt.subplots()
        figure.canvas.manager.set_window_title('Calibration Graph')

        # x axis
        plt.xlabel("Confidence")
        axis.set_xlim(0, 1)
        plt.xticks(confidence_bins + [1])

        # y axis
        plt.ylabel("Accuracy")
        axis.set_ylim(0, 1)
        plt.yticks([accuracy / 100 for accuracy in range(0, 101, 10)])

        # Data
        calibration_line, = plt.plot(
            [0, 1],
            [0, 1],
            "g--",
            label="Ideal calibration"
        )
        accuracy_bars = plt.bar(
            confidence_bins,
            accuracy_bins,
            figure=figure,
            width=0.05,
            label="Actual calibration"
        )

        # Supporting visualisations
        plt.grid(visible=True, axis="both")
        plt.legend(handles=[calibration_line, accuracy_bars], loc="upper left")

        return figure

    def save_calibration_graph(self, calibration_graph: plt.Figure) -> None:
        """
        Saves the calibration graph to the results folder as "calibration_graph.png".

        Note: If the figure is shown in a blocking manner before hand,
        an empty image would be saved.

        Args:
            calibration_graph (plt.Figure): The calibration graph to save
        """
        save_path = os.path.join(self.results_folder, "calibration_graph.png")
        calibration_graph.savefig(save_path)
        print(f"Saved calibration graph to {save_path}")

    def show_calibration_graph(self, calibration_graph: plt.Figure) -> None:
        """
        Shows the calibration graph to the results folder as "calibration_graph.png".

        Args:
            calibration_graph (plt.Figure): The calibration graph to show
        """
        calibration_graph.show()
        print("Displaying calibration graph")

    def generate_calibration_error(self) -> None:
        """
        Generate all the values need for calibration error.
        """
        print("Generating calibration error details.")
        average_confidence_bins, accuracy_bins, bin_count = (
            self.get_calibration_bins()
        )

        expected_calibration_error = torch.sum(
            (average_confidence_bins - accuracy_bins).abs() *
            bin_count / bin_count.sum()
        )
        maximum_calibration_error = torch.max(
            (average_confidence_bins - accuracy_bins).abs()
        )
        print(f"Expected calibration error: {expected_calibration_error}")
        print(f"Maximum calibration error: {maximum_calibration_error}")

        calibration_graph = self.get_calibration_graph(accuracy_bins)
        self.save_calibration_graph(calibration_graph)
        self.show_calibration_graph(calibration_graph)

    # General
    def print_separator(self) -> None:
        """
        Print a separator in the console.
        """
        print("\n---\n")

    def wait_from_all_windows(self) -> None:
        """
        Wait for all the pyplot windows to be closed before continuing,
        """
        if len(plt.get_fignums()) > 0:
            # Create a dummy window that blocks
            print("Waiting for all windows to be closed.")
            plt.show(block=True)
            plt.close()
            # Waiting for all existing windows to close before continuing

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
        self.display_confusion_matrix()
        self.print_separator()

        # Calibration error
        self.generate_calibration_error()
        self.print_separator()

        # Block till all opened windows are closed.
        print("Evaluation completed.")
        self.wait_from_all_windows()
        print("All windows closed. Exiting.")
