from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import tensorflow as tf
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops

import torchvision.models.detection.roi_heads as rh
from torchvision.models.detection.roi_heads import project_masks_on_boxes

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sobel_h = torch.tensor([[-1.,-2.,-1.],
                        [ 0., 0., 0.],
                        [ 1., 2., 1.]])

sobel_v = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]])

sobel_h_weights = sobel_h.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
sobel_v_weights = sobel_v.view(1, 1, 3, 3).repeat(1, 1, 1, 1)

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

    def __init__(self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
        custom_loss=False, 
        use_accessory=False
    ) :
        super().__init__(box_roi_pool,box_head,box_predictor, 
                    fg_iou_thresh,bg_iou_thresh,batch_size_per_image,positive_fraction,bbox_reg_weights,
                    score_thresh,nms_thresh,detections_per_img,mask_roi_pool,mask_head,mask_predictor,keypoint_roi_pool,
                    keypoint_head,keypoint_predictor)
        
        self.custom_loss = custom_loss
        self.use_accessory = use_accessory

    def custom_postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        accessory_scores # type: Tensor
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
            Also in that case we can replicate the same operations made for the boxes
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_accessories = torch.sigmoid(accessory_scores)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        accessories_list = pred_accessories.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_accessories = []
        for boxes, scores, image_shape, accessories in zip(pred_boxes_list, pred_scores_list, image_shapes, accessories_list):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            accessories = accessories.repeat(1,num_classes)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            accessories = accessories[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            accessories = accessories.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels, accessories = boxes[inds], scores[inds], labels[inds], accessories[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, accessories = boxes[keep], scores[keep], labels[keep], accessories[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, accessories = boxes[keep], scores[keep], labels[keep], accessories[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_accessories.append(accessories)

        return all_boxes, all_scores, all_labels, all_accessories

    def custom_assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, gt_accessories):
        # type: (List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
            What we want to do in here is to assign an accessory gt to every proposal, that's the only difference wrt the default function
        """
        matched_idxs = []
        labels = []
        accessories = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_acc_in_image in zip(proposals, gt_boxes, gt_labels, gt_accessories):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
                acc_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)
                # the labels positions are the same for gt_labels_in_image and gt_acc_in_image
                # because the way they were defined inside the Custom DataLoader...
                # so we can make the exact same operations of both
                acc_in_image = gt_acc_in_image[clamped_matched_idxs_in_image]
                acc_in_image = acc_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0
                acc_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler
                acc_in_image[ignore_inds] = -1

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            accessories.append(acc_in_image)
        return matched_idxs, labels, accessories

    def custom_select_training_samples(
        self,
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_accessories = [t["accessory"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels, accessories = self.custom_assign_targets_to_proposals(proposals, gt_boxes, gt_labels, gt_accessories)

        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            # same exact operations even in here
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            accessories[img_id] = accessories[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)

        return proposals, matched_idxs, labels, regression_targets, accessories

    def custom_fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets, accessory_scores, accessory_targets) :
        # type: (Tensor, Tensor, List[Tensor], List[Tensor], Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        """
        Computes the loss for Faster R-CNN.

        Args:
            class_logits (Tensor)
            box_regression (Tensor)
            labels (list[BoxList])
            regression_targets (Tensor)

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        labels = torch.cat(labels, dim=0)
        accessory_targets = torch.cat(accessory_targets, dim=0)

        regression_targets = torch.cat(regression_targets, dim=0)
        # cross_entropy considers all the classes loss and output them as a whole
        classification_loss = F.cross_entropy(class_logits, labels)
        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

        # for the binary classification purpose a binary_cross_entropy is needed
        # https://stackoverflow.com/questions/53628622/loss-function-its-inputs-for-binary-classification-pytorch

        accessory_predictions = F.sigmoid(accessory_scores)
        accessory_loss = F.binary_cross_entropy(accessory_predictions, accessory_targets.unsqueeze(1).to(torch.float32))

        return classification_loss, box_loss, accessory_loss

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
            if self.use_accessory :
                proposals, matched_idxs, labels, regression_targets, accessory_targets = self.custom_select_training_samples(proposals, targets)
            else :
                proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None
            accessory_targets = None

        # forward calls
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        if self.use_accessory :
            class_logits, box_regression, accessory_score = self.box_predictor(box_features)
        else :
            class_logits, box_regression = self.box_predictor(box_features)
        
        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")

            if self.use_accessory : 
                loss_classifier, loss_box_reg, loss_accessory = self.custom_fastrcnn_loss(class_logits, box_regression, labels, regression_targets, accessory_score, accessory_targets)
                losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg, "accessory_loss": loss_accessory}
            else :
                loss_classifier, loss_box_reg = rh.fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
                losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            if self.use_accessory :
                boxes, scores, labels, accessories = self.custom_postprocess_detections(class_logits, box_regression, proposals, image_shapes, accessory_score)
            else :
                boxes, scores, labels = rh.postprocess_detections(class_logits, box_regression, proposals, image_shapes)

            num_images = len(boxes)
            for i in range(num_images):
                if self.use_accessory :
                    result.append(
                        {
                            "boxes": boxes[i],
                            "labels": labels[i],
                            "scores": scores[i],
                            "accessories": accessories[i],
                        }
                    )
                else :
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

                if self.custom_loss : # edge agreement loss
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
