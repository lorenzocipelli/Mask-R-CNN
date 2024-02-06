import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from PIL import Image
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from pycocotools.coco import COCO

class ModaNetDataset(torch.utils.data.Dataset):
    """
        This class allows for loading the dataset and use it when needed
        to train your network (or finetune it). Redifined __init__, __getitem__ and
        __len__ to handle all the possible use cases of a dataset.
        To do it I followed the pytorch tutorial regarding Dataset class definition:

        https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    """

    def __init__(self, img_path, transforms):
        """
            Passing the image path and the transformation. Automatically list all the images
            and the images IDs using pycocotools for the latter
        """
        self.img_path = img_path
        self.transforms = transforms

        self.annotations_path = os.path.join(img_path, "modanet2018_instances_train.json")
        self.annotations = COCO(annotation_file=self.annotations_path)

        self.imgs = list(sorted(os.listdir(os.path.join(img_path, "images_train"))))
        self.imgs_ids = list(sorted(self.annotations.getImgIds())) # from COCO class
    
    def __getitem__(self, idx):
        """
            This method must return two elements (tuple), the image itself and all the annotations
            meaning that inside the annotations one can find the mask
        """

        # FIRST PART: load image

        img_name = self.imgs[idx]
        img_id = self.imgs_ids[idx] # used later for the annotations

        print("image_id: " + str(img_id))
        
        img_path = os.path.join(self.img_path, "images_train", self.imgs[idx])
        img = read_image(img_path)

        # print the image if needed
        #image = np.array(Image.open(os.path.join(self.img_path, "images_train", img_name)))        

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
        boxes, masks, labels = [], [], []
        #fig, ax = plt.subplots() # to delete
        for ann in img_anns : # for every annotation got for the image
            masks.append(torch.from_numpy(self.annotations.annToMask(ann))) # we got the mask in binary 2D array
            
            x1 = ann["bbox"][0]
            y1 = ann["bbox"][1]
            x2 = x1 + ann["bbox"][2]
            y2 = y1 + ann["bbox"][3]
            
            boxes.append([x1,y1,x2,y2]) 
            #bb = patches.Rectangle((ann["bbox"][0],ann["bbox"][1]), ann["bbox"][3],ann["bbox"][2], linewidth=2, edgecolor="blue", facecolor="none")
            #ax.add_patch(bb)
            labels.append(ann["category_id"]) # get the cat from annotation

        #ax.imshow(image)
        #plt.show()
            
        # THIRD PART: put everything inside img and target
            
        img = tv_tensors.Image(img)

        masks = torch.as_tensor(np.asarray(masks), dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # for every BB compute the area: (y2-y1) * (x2-x1) 
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(img_anns),), dtype=torch.int64)
            
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYWH", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        print(img)
        print(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)