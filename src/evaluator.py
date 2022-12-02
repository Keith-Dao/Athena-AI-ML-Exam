"""
    Dataset evaluator module.
"""
import torch
from torch.utils.data import DataLoader
import torchvision


class Evaluator:
    """
    Evaluator for a given dataset and model.
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
        self.dataset = torchvision.datasets.ImageFolder(
            dataset_path, transform=data_transform()
        )
        self.loader = DataLoader(self.dataset, batch_size=32, shuffle=True)

        # Model
        self.model = torchvision.models.get_model(
            f"convnext_{model_size}", weights="DEFAULT"
        ).to(self.device)
