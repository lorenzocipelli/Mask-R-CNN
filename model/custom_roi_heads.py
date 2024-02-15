from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor

import torchvision.models.detection.roi_heads as rh
from torchvision.models.detection.roi_heads import project_masks_on_boxes

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def l2_loss(y_true, y_pred):
        return torch.mean(torch.pow(torch.abs(y_pred - y_true), 2), axis=-1)

# new loss -> edge agreement: L_edge
def edge_agreement_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs) :
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
        From: https://arxiv.org/pdf/1809.07069.pdf

        This method applies, for each of the outputs of the Mask Head, a Sobel filtering on the GT 28x28 mask
        corresponding to the predicted 28x28 output of the head, and on the output aswell. After this computation
        the custom loss is established by the generalized power mean of the absolute difference between 
        the target y_hat and the prediction y
        We need the same arguments as for the maskrcnn_loss

        Args:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
    """

    sobel_h = torch.tensor([[-1.,-2.,-1.],
                            [ 0., 0., 0.],
                            [ 1., 2., 1.]])
    
    sobel_v = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]])

    sobel_h_weights = sobel_h.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
    sobel_v_weights = sobel_v.view(1, 1, 3, 3).repeat(1, 1, 1, 1)

    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]

    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    total_edge_loss = 0
    
    for idx in range(labels.shape[0]) :
        gt_label = labels[idx]
        # we need to compute edge detection on both GT and predicted mask
        pred_mask = mask_logits[idx][gt_label].unsqueeze(0).unsqueeze(0)
        gt_mask = mask_targets[idx].unsqueeze(0).unsqueeze(0)

        sobel_h_output = F.conv2d(pred_mask.to(DEVICE), sobel_h_weights.type(torch.cuda.FloatTensor).to(DEVICE))
        filtered_pred = F.conv2d(sobel_h_output.to(DEVICE), sobel_v_weights.type(torch.cuda.FloatTensor).to(DEVICE))

        sobel_h_output = F.conv2d(gt_mask.to(DEVICE), sobel_h_weights.type(torch.cuda.FloatTensor).to(DEVICE))
        filtered_gt = F.conv2d(sobel_h_output.to(DEVICE), sobel_v_weights.type(torch.cuda.FloatTensor).to(DEVICE))

        pixel_wise_edge_loss = 0
        if filtered_gt.size()[0] > 0 :
            pixel_wise_edge_loss = l2_loss(filtered_gt, filtered_pred)

        # return the mean of the pixelwise edge agreement loss
        edge_loss = torch.mean(pixel_wise_edge_loss)
        total_edge_loss += edge_loss

    return total_edge_loss / labels.shape[0]

class CustomRoIHeads(rh.RoIHeads):
    """
        The only difference between this class and the original RoiHeads from PyTorch
        is that this class in the forward method does return, other than the classification loss and
        the BB loss, it returns also the new Edge Agreement Loss
        This loss is implemented just for Image Segmentation, meaning that we want to predict a mask for 
        a predicted object returned from the possible RoIs. That's why the new loss is not implemented for
        the keypoint predictor
    """

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = rh.fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            loss_edge_agreement = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = rh.maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}

                # edge agreement loss :
                # compute the loss with respect to the selected mask and the ground truth mask
                loss_edge_agreement = edge_agreement_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_edge_agreement = {"loss_edge_agreement": loss_edge_agreement}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = rh.maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask) # add mask loss to the losses dictionary
            losses.update(loss_edge_agreement) # add edge agreement loss to the losses dictionary

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                if targets is None or pos_matched_idxs is None:
                    raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = rh.keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                if keypoint_logits is None or keypoint_proposals is None:
                    raise ValueError(
                        "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                    )

                keypoints_probs, kp_scores = rh.keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
            losses.update(loss_keypoint)

        return result, losses
