import json
import os

import matplotlib.pyplot as plt
from torchvision.io import read_image

from utils.modanet_dataset import ModaNetDataset
from utils.utils import get_transform

print('getcwd:      ', os.getcwd())

modanet = ModaNetDataset("dataset", get_transform())

modanet[50]
