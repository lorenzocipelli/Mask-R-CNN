import torchvision
import torch.nn as nn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_mask_rcnn_model(num_classes, args) :
    if args.pretrained :
        if args.version == "V1":
            # load an instance segmentation model pre-trained on COCO
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        elif args.version == "V2":
            # load an instance segmentation model pre-trained on COCO
            model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    else :
        if args.version == "V1":
            # load an instance segmentation model pre-trained on COCO
            model = torchvision.models.detection.maskrcnn_resnet50_fpn()
        elif args.version == "V2":
            # load an instance segmentation model pre-trained on COCO
            model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2()

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

class MaskRCNN(nn.Module) :
    def __init__(self, num_classes, args):
        super().__init__()
        self.model = get_mask_rcnn_model(num_classes, args)

    def forward(self, images, targets=None):
        return self.model(images,targets)