import torch

import os
import shutil

from os import listdir
from PIL import Image
from torchvision.transforms import v2 as T

def collate_fn(batch):
   return tuple(zip(*batch)) 

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    with open(path, 'rb') as f:
        try:
            img = Image.open(f)
            img = img.convert('RGB')
            return img
        except :
            print("Immagine corrotta trovata: " + str(path))

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

def spot_corrupted_files() :
    broken_img_list = []
    path = 'dataset/images_train'
    for filename in listdir(path):
        if filename.endswith('.jpg'):
            img_path = os.path.join(path, filename)
            try:
                img = Image.open(img_path) # open the image file
                img.verify() # verify that it is, in fact an image
            except (IOError, SyntaxError) as e:
                broken_img_list.append(img_path)
                print('Bad file:', img_path) # print out the names of corrupt files

    print("We have " + str(len(broken_img_list)) + " broken images")

    # move to broken file folder and delete from main folder
    for broken_img in broken_img_list :
        move_remove(broken_img)

    print("Moving Broken Files Completed !")

def move_remove(path_img) :
    shutil.copy(path_img, 'dataset/broken_images_train/')
    os.remove(path_img)