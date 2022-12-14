"""
    Dataset inferencer module.
"""
import os
import shutil

import torch
import torchvision
from tqdm import tqdm

from src.image_folder_dataset import ImageFolderDataset


class Inferencer:
    """
    Inferencer for a given dataset and model.
    """

    default_model_choice = "tiny"
    model_choices = ["tiny", "small", "base", "large"]

    def __init__(self, dataset_path: str, model_size: str) -> None:
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # Data
        data_transform = torchvision.models.get_weight(
            f"ConvNeXt_{model_size.capitalize()}_Weights.DEFAULT"
        ).transforms
        self.dataset = ImageFolderDataset(
            dataset_path, transform=data_transform()
        )

        # Model
        self.model = torchvision.models.get_model(
            f"convnext_{model_size}",
            num_classes=len(self.dataset.classes)
        ).to(self.device)
        self.model.eval()  # No training is performed here.

        # Inference data
        self.confidences = []
        self.predictions = []

    def infer(self, save_path: str | None = None):
        """
        Perform inference on the provided data and save all false positives to the provided path.

        Note: It is assumed that the class folders have already been created beforehand.

        Args:
            save_path (str | None): Path to save all the false positive images
        """
        print("Performing inference on the provided dataset.")
        for data, label, source_path in tqdm(self.dataset):
            data = data.unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(data)
                softmax = torch.exp(logits) / torch.sum(torch.exp(logits))
                output = torch.max(softmax.to("cpu"), dim=-1)

            confidence = output.values.item()
            self.confidences.append(confidence)

            prediction = output.indices.item()
            self.predictions.append(prediction)
            if save_path and prediction != label:
                shutil.copy(
                    source_path,
                    os.path.join(
                        save_path,
                        self.dataset.classes[prediction],
                        os.path.basename(source_path)
                    )
                )

        print("Inference completed.")

    def get_class_labels(self) -> list[str]:
        """
        Gets the class labels.

        Returns:
            List of the class labels in the same order as the index
        """
        return self.dataset.classes

    def get_true_labels(self) -> list[int]:
        """
        Get the true labels.

        Returns:
            The true labels in the same order as inference.
        """
        return self.dataset.targets

    def get_predicted_labels(self) -> list[int]:
        """
        Get the predicted labels.

        Returns:
            The predicted labels in the same order as inference.
        """
        return self.predictions

    def get_confidences(self) -> list[float]:
        """
        Get the confidence levels.

        Returns:
            The confidence levels in the same order as inference.
        """
        return self.confidences
