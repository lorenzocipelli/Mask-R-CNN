import json
import os
import torchvision
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.utils import collate_fn
from model.mask_rcnn import MaskRCNN

from torchvision.io import read_image
from torch.utils.data import DataLoader
from utils.modanet import ModaNetDataset
from utils.utils import get_transform
from engine import Engine

# print('getcwd: ', os.getcwd()) # working directory

FIRST_STOP = 32728 # 70 %
SECOND_STOP = 7013 # 15 %

def get_subsets(dataset) :
    # total -> 46754 elements (100 %)
    # train -> 32728 elements (70 %)
    # validation -> 7013 elements (15 %)
    # test -> 7013 elements (15 %)

    idxs = torch.randperm(len(dataset)).tolist()
    # modificare di nuovo, messo a mille per ridurre l'uso di memoria durante lo sviluppo
    train_list, valid_list, test_list = idxs[:FIRST_STOP], idxs[FIRST_STOP:FIRST_STOP+SECOND_STOP], idxs[-SECOND_STOP:]
    train = torch.utils.data.Subset(dataset, train_list)
    valid = torch.utils.data.Subset(dataset, valid_list)
    test = torch.utils.data.Subset(dataset, test_list)

    return train, valid, test

def main() :
    modanet = ModaNetDataset("dataset", get_transform())

    train_modanet, val_modanet, test_modanet = get_subsets(modanet)

    train_loader = DataLoader(train_modanet, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)
    valid_loader = DataLoader(val_modanet, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_modanet, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)

    engine = Engine(train_loader, valid_loader, test_loader)

    engine.train()

if __name__ == "__main__" :
    main()