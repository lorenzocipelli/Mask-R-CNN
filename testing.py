import json
import os
import torchvision
import matplotlib.pyplot as plt

from model.mask_rcnn import MaskRCNN

from torchvision.io import read_image
from utils.modanet_dataset import ModaNetDataset
from utils.utils import get_transform
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

print('getcwd:      ', os.getcwd())

#modanet = ModaNetDataset("dataset", get_transform())

model = MaskRCNN(num_classes=14)