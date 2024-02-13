import torchvision
import torch.nn as nn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import roi_heads

from .custom_roi_heads import CustomRoIHeads

def get_mask_rcnn_model(num_classes, args) :
    if args.pretrained : # load an instance segmentation model pre-trained on COCO
        if args.version == "V1":        
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        elif args.version == "V2":
            model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    else :
        if args.version == "V1":
            model = torchvision.models.detection.maskrcnn_resnet50_fpn()
        elif args.version == "V2":
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

    old_roi_head = model.roi_heads
    new_roi_head = CustomRoIHeads(old_roi_head.box_roi_pool, old_roi_head.box_head, old_roi_head.box_predictor, 
                old_roi_head.proposal_matcher.high_threshold, old_roi_head.proposal_matcher.low_threshold, 
                old_roi_head.fg_bg_sampler.batch_size_per_image, old_roi_head.fg_bg_sampler.positive_fraction,
                old_roi_head.box_coder.weights, old_roi_head.score_thresh, old_roi_head.nms_thresh, old_roi_head.detections_per_img,
                old_roi_head.mask_roi_pool, old_roi_head.mask_head, old_roi_head.mask_predictor,
                old_roi_head.keypoint_roi_pool, old_roi_head.keypoint_head, old_roi_head.keypoint_predictor)
    
    model.roi_heads = new_roi_head

    return model

class MaskRCNN(nn.Module) :
    def __init__(self, num_classes, args):
        super().__init__()
        self.model = get_mask_rcnn_model(num_classes, args)

    def forward(self, images, targets=None):
        return self.model(images,targets)