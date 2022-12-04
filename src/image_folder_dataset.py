"""
    Image dataset loader extending torchvision's ImageFolder
"""
import torchvision


class ImageFolderDataset(torchvision.datasets.ImageFolder):
    """
    An extension on torchvision's ImageFolder to include the filepath when accessing the file.
    """

    def __getitem__(self, index: int):
        return *super().__getitem__(index), self.samples[index][0]
