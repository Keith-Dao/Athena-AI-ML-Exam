"""
    Dataset inferencer module.
"""
import torch
import torchvision

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
        model = torchvision.models.get_model(
            f"convnext_{model_size}", weights="DEFAULT"
        )
        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features, len(self.dataset.classes)
        )
        self.model = model.to(self.device)
