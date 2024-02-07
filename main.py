import json
import os
import torchvision
import argparse
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.utils import collate_fn
from model.mask_rcnn import MaskRCNN

from torchvision.io import read_image
from torch.utils.data import DataLoader
from utils.modanet import ModaNetDataset
from utils.utils import get_transform

print('getcwd: ', os.getcwd())

def get_subsets(dataset) :
    # total -> 46.868 elements (100 %)
    # train -> 32808 elements (70 %)
    # validation -> 7030 elements (15 %)
    # test -> 7030 elements (15 %)

    idxs = torch.randperm(len(dataset)).tolist()
    train_list, valid_list, test_list = idxs[:32808], idxs[32808:32808+7030], idxs[-7030:]
    train = torch.utils.data.Subset(dataset, train_list)
    valid = torch.utils.data.Subset(dataset, valid_list)
    test = torch.utils.data.Subset(dataset, test_list)

    return train, valid, test

def main() :
    modanet = ModaNetDataset("dataset")#, get_transform())

    train_modanet, val_modanet, test_modanet = get_subsets(modanet)

    #all_dataset = DataLoader(modanet, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)

    train_loader = DataLoader(train_modanet, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)
    #valid_loader = DataLoader(val_modanet, batch_size=16, shuffle=False, num_workers=4)
    #test_loader = DataLoader(test_modanet, batch_size=16, shuffle=False, num_workers=4)

    """for i, data in enumerate(modanet, 0): 
        print("Indice Enumerate: " + str(i))
        #print(data)"""

    prog_bar = tqdm(train_loader, total=len(train_loader))

    for i, data in enumerate(prog_bar):
        j = 1
        #print("Indice Enumerate: " + str(i))
        #print("New target: ")
        #print(target) 

    print("fine")

    #model = MaskRCNN(num_classes=14)

if __name__ == "__main__":
    main()