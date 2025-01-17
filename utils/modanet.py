import os
import torch
import cv2
import numpy as np

from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from pycocotools.coco import COCO

ACCESSORIES_IDs = [1,2,7,12,13]

class ModaNetDataset(torch.utils.data.Dataset):
    """
        This class allows for loading the dataset and use it when needed
        to train your network (or finetune it). Redifined __init__, __getitem__ and
        __len__ to handle all the possible use cases of a dataset.
        To do it I followed the pytorch tutorial regarding Dataset class definition:

        https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    """

    def __init__(self, args, transforms):
        """
            Passing the image path and the transformation. Automatically list all the images
            and the images IDs using pycocotools for the latter
        """
        self.img_path = args.dataset_path
        self.transforms = transforms
        self.use_accessory = args.use_accessory

        self.annotations_path = os.path.join(self.img_path, "modanet2018_instances_train_fix.json")
        self.annotations = COCO(annotation_file=self.annotations_path)

        self.imgs = list(sorted(os.listdir(os.path.join(self.img_path, "images_train"))))
        self.imgs_ids = list(sorted(self.annotations.getImgIds())) # from COCO class
    
    def __getitem__(self, idx):
        """
            This method must return two elements (tuple), the image itself and all the annotations
            meaning that inside the annotations one can find the mask
        """

        # FIRST PART: load image

        img_id = self.imgs_ids[idx] # used later for the annotations

        #print("image_id: " + str(img_id))
        
        img_path = os.path.join(self.img_path, "images_train", self.imgs[idx])

        img = read_image(img_path)    

        # SECOND PART: load all the annotations

        # we get all the annotations IDs corresponding to the image ID
        ann_ids = self.annotations.getAnnIds(imgIds=[img_id])

        # we get all the annotations corresponding to the ann_ids computed from image ID
        img_anns = self.annotations.loadAnns(ann_ids)

        """
            The following is what the model expect as input from the variable target
            So we need to compute them for every annotation of the image

            target = {}
            target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
            target["masks"] = tv_tensors.Mask(masks)
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
        """
        
        target = {}
        boxes, masks, labels, new_accessory = [], [], [], []

        for ann in img_anns : # for every annotation got for the image
            mask = self.annotations.annToMask(ann) # we got the mask in binary 2D array
            mask = cv2.resize(mask, (400, 600), cv2.INTER_LINEAR) # from https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
            masks.append(mask)
            
            x1 = ann["bbox"][0]
            y1 = ann["bbox"][1]
            x2 = x1 + ann["bbox"][2]
            y2 = y1 + ann["bbox"][3]
            
            boxes.append([x1,y1,x2,y2])
            labels.append(ann["category_id"]) # get the category from annotation

            """
                Project Specification :
                You have to automatically generate the ground truth during the training following the rule: 
                    "True if class is one of the following: bag, belt, sunglasses, headwear, scarf tie. Otherwise is false".

                As suggested: this request implies the training of a binary classifier
                For every annotation inside one image we add 1 if this annotation is an accessory, 0 otherwise
                In this way we get a list of binary values
            """
            if self.use_accessory: # generate ground truth for accessory
                if ann['category_id'] in ACCESSORIES_IDs: 
                    new_accessory.append(1)
                else :
                    new_accessory.append(0)
            
        # THIRD PART: put everything inside img and target
            
        img = tv_tensors.Image(img)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # this area computation mathod works only if all images
        # have at least one BB, otherwhise it will crash
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd, boxes.shape[0] for MeanAveragePrecision computation
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        new_accessories = torch.as_tensor(new_accessory,  dtype=torch.float32) # kind of boolean array

        target["boxes"] = boxes
        target["masks"] = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([img_id]) # to allow v.to(DEVICE)
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.use_accessory:
            target["accessory"] = new_accessories

        if self.transforms is not None: # apply transforms if they are defined
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
    def show_info(self):
        print(self.imgs_ids)