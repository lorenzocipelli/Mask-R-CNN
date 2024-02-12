import torch
import json

import os
import shutil

from os import listdir
from PIL import Image
from torchvision.transforms import v2 as T
from pycocotools.coco import COCO

def collate_fn(batch):
   return tuple(zip(*batch)) 

def transforms_pipeline():
    """
        Trasnformations to be made for data augmentation. 
        The follwing link shows all the possible transformations
        that are available with the new v0.15.0 torchvision:
        
        https://pytorch.org/vision/stable/transforms.html

        This new API allows to easily write data augmentation pipelines
    """
    transforms = []

    transforms.append(T.ColorJitter())
    transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())

    return T.Compose(transforms)

def remove_from_json(to_remove) :
    data = {}
    print("Removing all these elements from JSON...")
    # open file in read-mode
    with open('dataset/modanet2018_instances_train.json', 'r') as file:
        data = json.load(file)
        #print(data["images"])

        for elem in to_remove :
            for img_idx in reversed(range(len(data["images"]))) :
                #print(data["images"][img_idx])
                if data["images"][img_idx]["id"] == elem :
                    del data["images"][img_idx]

    #print(data["images"])

    with open('dataset/modanet2018_instances_train_fix.json', 'w') as f:
        json.dump(data, f)

    print("Removed all Elements from JSON !")

def spot_imgs_without_annotations() :
    missing_annotation_img_list = []
    missing_annotation_ids = []
    path = 'dataset/images_train'

    annotations_path = os.path.join("dataset", "modanet2018_instances_train.json")
    annotations = COCO(annotation_file=annotations_path)

    imgs = list(sorted(os.listdir(os.path.join("dataset", "images_train"))))

    for img in imgs :
        img_id = int(img.lstrip('0')[:-4]) # cleaning ID from img
        ann_ids = annotations.getAnnIds(imgIds=[img_id])
        if len(ann_ids) < 1 :
            img_path = os.path.join(path, img)
            missing_annotation_img_list.append(img_path)
            missing_annotation_ids.append(img_id)

    print("We have " + str(len(missing_annotation_img_list)) + " images without annotations")
        
    for missing_annotation_image in missing_annotation_img_list :
        move_remove(missing_annotation_image)

    print("Moving Images without Annotations Completed !")

    return missing_annotation_ids

def spot_corrupted_files() :
    broken_img_list = []
    broken_img_ids = []
    path = 'dataset/images_train'
    for filename in listdir(path):
        if filename.endswith('.jpg'):
            img_path = os.path.join(path, filename)
            try:
                img = Image.open(img_path) # open the image file
                img.verify() # verify that it is, in fact an image
            except (IOError, SyntaxError) as e:
                broken_img_list.append(img_path)
                broken_img_ids.append(int(filename.lstrip('0')[:-4]))
                print('Bad file:', img_path) # print out the names of corrupt files

    print("We have " + str(len(broken_img_list)) + " broken images")

    # move to broken file folder and delete from main folder
    for broken_img in broken_img_list :
        move_remove(broken_img)

    print("Moving Broken Files Completed !")

    return broken_img_ids

def move_remove(path_img) :
    shutil.copy(path_img, 'dataset/broken_images_train/')
    os.remove(path_img)

def clean_dataset() :
    img_to_remove = spot_corrupted_files()
    img_to_remove += spot_imgs_without_annotations()
    remove_from_json(img_to_remove)
 
#clean_dataset()