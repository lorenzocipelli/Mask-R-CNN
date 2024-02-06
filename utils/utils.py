import torch

from torchvision.transforms import v2 as T


def get_transform():
    """
        Trasnformations to be made for data augmentation. 
        The follwing link shows all the possible transformations
        that are available with the new v0.15.0 torchvision:
        
        https://pytorch.org/vision/stable/transforms.html

        This new API allows to easily write data augmentation pipelines
    """
    transforms = []

    transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())

    return T.Compose(transforms)