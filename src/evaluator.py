"""
    Dataset evaluator module.
"""
from torch.utils.data import DataLoader
import torchvision


class Evaluator:
    """
    Evaluator for a given dataset and model.
    """

    model_size_to_model = {
        "tiny": (
            torchvision.models.convnext_tiny,
            torchvision.models.ConvNeXt_Tiny_Weights,
        ),
        "small": (
            torchvision.models.convnext_small,
            torchvision.models.ConvNeXt_Small_Weights,
        ),
        "base": (
            torchvision.models.convnext_base,
            torchvision.models.ConvNeXt_Base_Weights,
        ),
        "large": (
            torchvision.models.convnext_large,
            torchvision.models.ConvNeXt_Large_Weights,
        ),
    }
    default_model_choice = "tiny"
    model_choices = list(model_size_to_model.keys())

    def __init__(self, dataset_path: str, model_size: str) -> None:
        model, weights = Evaluator.model_size_to_model[model_size]
        self.model = model(weights=weights.DEFAULT)
        self.dataset = torchvision.datasets.ImageFolder(
            dataset_path, transform=weights.DEFAULT.transforms()
        )
        self.data_iter = DataLoader(self.dataset, batch_size=32, shuffle=True)
