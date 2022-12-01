"""
    Dataset evaluator module.
"""
import torchvision


class Evaluator:
    """
    Evaluator for a given dataset and model.
    """

    # Dataset
    dataset_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    # Model
    model_size_to_model = {
        "tiny": torchvision.models.convnext_tiny,
        "small": torchvision.models.convnext_small,
        "base": torchvision.models.convnext_base,
        "large": torchvision.models.convnext_large,
    }

    def __init__(self, dataset_path: str, model_size: str) -> None:
        # Dataset
        self.dataset = torchvision.datasets.ImageFolder(
            dataset_path, transform=Evaluator.dataset_transform
        )

        # Model
        self.model = Evaluator.model_size_to_model[model_size](
            weights="DEFAULT"
        )
