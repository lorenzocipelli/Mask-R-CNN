import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import math
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from model.mask_rcnn import MaskRCNN
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

COLORS = [(51, 153, 255), (255, 102, 0), (0, 204, 153), (153, 102, 255), (255, 255, 0),
          (204, 102, 153), (204, 102, 0), (51, 51, 0), (0, 153, 0), (51, 51, 255)]

COLORS = COLORS + COLORS # double the number of colors, even if they repeat, to be sure when printing

CLASSES = ["bag","belt","boots","footwear",
           "outer","dress","sunglasses",
           "pants","top","shorts","skirt",
           "headwear","scarf & tie",]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_TO_PRINT = 10
IMGS_PER_ROW = 5

# Enable cuDNN auto-tuner:
# Autotuner runs a short benchmark and selects the kernel 
# with the best performance on a given hardware for a given input size
torch.backends.cudnn.benchmark = True
# PyTorch by default does not disable the autograd and other profilers while training
# To enhance speed while training it is recommended to turn them off manually.
def set_debug_apis(state: bool = False):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)

class Engine() :
    """
        Class created for implementing automated
        testing and training
    """

    def __init__(self, train_load, valid_load, test_load, args) :
        self.model_name = args.model_name
        self.epochs = args.epochs
        self.train_loader = train_load
        self.valid_loader = valid_load
        self.test_loader = test_load
        self.model_name = args.model_name
        self.check_path = args.saving_path
        self.use_amp = args.use_amp
        self.use_accessory = args.use_accessory
        self.custom_loss = args.custom_loss
        self.num_classes = 14 # background has to be considered as the first class, so we add one more

        # qui andrà modificato con 14 fisso, l'ultima classe non serve a nulla
        self.model = MaskRCNN(num_classes=self.num_classes, args=args).to(DEVICE)

        if args.resume :
            self.load_model(resume=args.resume_name)

        # The call to model.parameters()
        # in the SGD constructor will contain the learnable parameters (defined 
        # with torch.nn.Parameter) which are members of the model
        if args.opt == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)
        elif args.opt == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        if args.mode == "hyper_tuning" :
            self.run_name = "/hpc/home/lorenzo.cipelli/mask-rcnn/runs/" + self.model_name
            self.tuning = True
        else :
            self.tuning = False

        if self.tuning :
            self.summary = SummaryWriter(log_dir=self.run_name)
        else :
            self.summary = SummaryWriter(log_dir=f'/runs/modanet') # will write in ./runs/ folder 

    def save_model(self, epoch, iteration=0) :
        # if you want to save the model
        checkpoint_name = str(self.model_name) + "_epoch" + str(epoch) + "_" + str(iteration) + "_" + "mask_r_cnn"
        check_path = os.path.join(self.check_path, checkpoint_name)
        torch.save(self.model.state_dict(), check_path)
        print("Model saved!")

    def load_model(self, resume=""):
        # function to load the model
        if resume != "" :
            check_path = os.path.join(self.check_path, resume)
            self.model.load_state_dict(torch.load(check_path, map_location=torch.device(DEVICE)))
            print("Model loaded!", flush=True)
        else :
            check_path = os.path.join(self.check_path, self.model_name + "_epoch" + str(0) + "_" + "final" + "_" + "mask_r_cnn")
            self.model.load_state_dict(torch.load(check_path, map_location=torch.device(DEVICE)))
            print("Model loaded!", flush=True)

    def train(self) :
        print("Training Begins !")
        self.model.train()
        num_elements_train = len(self.train_loader)

        set_debug_apis(state=False)
        if self.use_amp :
            # initialize gradient scaler
            scaler = torch.cuda.amp.GradScaler() 

        for epoch in range(self.epochs):
            print("Training of epoch " + str(epoch) + " Begins !")
            prog_bar = tqdm(self.train_loader, total=num_elements_train)
            """ 
                Our model compute the losses for every head, and return them in a dictionary 
                like the following :
                {'loss_classifier': tensor(2.8990, device='cuda:0', grad_fn=<NllLossBackward0>),
                 'loss_box_reg': tensor(0.3307, device='cuda:0', grad_fn=<DivBackward0>),
                 'loss_mask': tensor(1.2220, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>),
                 'loss_objectness': tensor(0.3347, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>),
                 'loss_rpn_box_reg': tensor(0.0507, device='cuda:0', grad_fn=<DivBackward0>) } 

                 optionally even the edge agreement loss 
            """
            
            loss_dict = {
                    'loss_classifier': 0,
                    'loss_box_reg': 0,
                    'loss_mask': 0,
                    'loss_objectness': 0,
                    'loss_rpn_box_reg': 0
                }
            if self.use_accessory :
                loss_dict.update({'accessory_loss': 0})
            if self.custom_loss : # optionally add key, value for edge agreement loss
                loss_dict.update({'loss_edge_agreement': 0})                 

            running_loss = 0.0
            running_loss_dict = loss_dict
            
            for i, data in enumerate(prog_bar):
                images, targets = data
                # from .....torchvision_tutorial.html
                images =  list(image.to(DEVICE) for image in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                if self.use_amp :
                    # instances of autocast serve as context managers or decorators 
                    # that allow regions of the script to run in mixed precision
                    with torch.cuda.amp.autocast(): 
                        loss_dict = self.model(images, targets)
                else :
                    loss_dict = self.model(images, targets)

                # from https://github.com/pytorch/vision/blob/main/references/detection/engine.py
                # If you have multiple losses you can sum them and then call backwards once
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")
                    print(loss_dict)
                    sys.exit(1)
                
                if self.use_amp : # run in mixed precision
                    scaler.scale(losses).backward() # compute gradients
                    scaler.step(self.optimizer) # update parameters of the model
                    scaler.update()
                
                else :
                    losses.backward() # compute gradients
                    self.optimizer.step() # update parameters of the model

                # print statistics
                running_loss += loss_value
                for loss in loss_dict :
                    running_loss_dict[loss] += loss_dict[loss]

                if i % 200 == 199: # save the result into the SummaryWriter every 200 mini-batch
                    self.summary.add_scalar("overall_loss_training", running_loss / 200, epoch * num_elements_train + i)
                    for field_running_loss in running_loss_dict : # saving all training losses that the model outputs
                        self.summary.add_scalar(field_running_loss, running_loss_dict[field_running_loss] / 200, epoch * num_elements_train + i)

                    print(f'[it: {i + 1}] loss: {running_loss / 200:.3f}')
                    running_loss = 0.0 # reset the running loss overall value
                    # reset the running loss dict
                    running_loss_dict = {
                        'loss_classifier': 0,
                        'loss_box_reg': 0,
                        'loss_mask': 0,
                        'loss_objectness': 0,
                        'loss_rpn_box_reg': 0
                    }
                    if self.use_accessory :
                        running_loss_dict.update({'accessory_loss': 0})
                    if self.custom_loss : # optionally add key, value for edge agreement loss
                        running_loss_dict.update({'loss_edge_agreement': 0})

                #if i % 1000 == 999: # every 1000 mini-batch save the model
                #    self.save_model(epoch, i)

            self.save_model(epoch)

            # at the end of every epoch work on validation set
            # The validation set is just used to give an approximation of 
            # generalization error at any epoch but also, crucially, 
            # for hyperparameter optimization.

            self.validate(epoch) # pass the epoch for SummaryWriter of validation loss

        print("Training Finished !")

        if not(self.tuning) :
            self.evaluate() # compute the mAP metrics over the validation set

        self.summary.close()

    def validate(self, epoch) :
        print("Validation of epoch " + str(epoch) + " begins !")
        num_elements_validation = len(self.valid_loader)
        prog_bar = tqdm(self.valid_loader, total=num_elements_validation)
        running_loss = 0.0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(prog_bar):
                images, targets = data
                # from .....torchvision_tutorial.html
                images =  list(image.to(DEVICE) for image in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                if self.use_amp :
                    with torch.cuda.amp.autocast(): 
                        loss_dict = self.model(images, targets)
                else :
                    loss_dict = self.model(images, targets)

                # from https://github.com/pytorch/vision/blob/main/references/detection/engine.py
                # If you have multiple losses you can sum them and then call backwards once
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                # print statistics
                running_loss += loss_value

        # save the result into the SummaryWriter  at the end of validation
        self.summary.add_scalar("overall_loss_validation", running_loss / num_elements_validation, epoch)
        print("Validation of epoch " + str(epoch) + " finished !")
    
    def test(self, want_to_print=12) :
        if want_to_print > MAX_TO_PRINT :
            want_to_print = MAX_TO_PRINT # limit the value if necessary
        elif want_to_print <= 0 :
            want_to_print = 1
        
        output_images_list = []
        not_predicted = 0 # count for images without predictions, used in when printing in a grid
        # set it to evaluation mode
        self.model.eval()
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            prog_bar = tqdm(self.test_loader, total=want_to_print)
            for i, data in enumerate(prog_bar):
                if i >= want_to_print : # stop when we have the requested number of images
                    break

                images, targets = data

                images =  list(image.to(DEVICE) for image in images)

                predictions = self.model([images[0]])
                pred = predictions[0]

                image = (255.0 * (images[0] - images[0].min()) / (images[0].max() - images[0].min())).to(torch.uint8)
                image = image[:3, ...]

                if self.use_accessory :
                    preds = [(f"{CLASSES[label-1]}: {score:.3f} - acc: {round(acc.item())}", boxes, masks) 
                                for label, score, boxes, acc, masks in zip(
                                    pred["labels"].detach().cpu(),
                                    pred["scores"].detach().cpu(),
                                    pred["boxes"].detach().cpu(),
                                    pred["accessories"].detach().cpu(),
                                    pred["masks"].detach().cpu()) if score >= 0.65]
                else : 
                    preds = [(f"{CLASSES[label-1]}: {score:.3f}", boxes, masks) 
                                for label, score, boxes, masks in zip(
                                    pred["labels"].detach().cpu(),
                                    pred["scores"].detach().cpu(),
                                    pred["boxes"].detach().cpu(),
                                    pred["masks"].detach().cpu()) if score >= 0.65]    

                if len(preds) == 0 :
                    not_predicted += 1
                    print("\nNo prediction was found for this image")
                else :
                    pred_labels, pred_boxes, pred_masks = [], [], []
                    for (label, box, mask) in preds :
                        pred_labels.append(label)
                        pred_boxes.append(np.array(box))
                        pred_masks.append(np.array(mask))

                    pred_boxes = torch.tensor(pred_boxes, dtype=torch.float16).long()
                    pred_masks = (torch.tensor(pred_masks, dtype=torch.float16) > 0.5).squeeze(1)

                    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors=COLORS)
                    output_image = draw_segmentation_masks(output_image, pred_masks, alpha=0.6, colors=COLORS)
                    output_images_list.append(output_image)

            # the number of element for each row is fixed (number of columns)
            # the number of columns depends on the number of images one want to print
            num_rows = math.ceil(want_to_print/IMGS_PER_ROW)
            f, axarr = plt.subplots(num_rows, IMGS_PER_ROW)

            for x in range(num_rows) :
                for y in range(IMGS_PER_ROW) :
                    index = x*IMGS_PER_ROW + y
                    if index > (len(output_images_list) - not_predicted) :
                        break
                    else :
                        axarr[x,y].imshow(output_images_list[x*IMGS_PER_ROW + y].permute(1, 2, 0))

            f.set_figheight(10)
            f.set_figwidth(19)
            f.tight_layout()
            plt.show()

    def evaluate(self) :
        """
            The evalutation metric choosen is mAP, Mean Average Precision.
            That is because COCO utilizes Average Precision to compare different models performances
            on the detection task. The Mean Average Precision is just the mean computed over all
            the average precisions for each class of the dataset
        """

        """
            From: https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html

            The average precision is defined as the area under the precision-recall curve. 
            For object detection the recall and precision are defined based on the intersection of union (IoU) 
            between the predicted bounding boxes and the ground truth bounding boxes e.g. if two boxes have an IoU > t 
            (with t being some threshold) they are considered a match and therefore considered a true positive. 
            The precision is then defined as the number of true positives divided by the number of all detected 
            boxes and the recall is defined as the number of true positives divided by the number of all ground boxes.
            
            How to compute mAP:
                1)  Generate the prediction scores using the model
                2)  Convert the prediction scores to class labels
                3)  Calculate the confusion matrix: TP, FP, TN, FN
                4)  Calculate the precision and recall metrics
                5)  Calculate the area under the precision-recall curve
                6)  Measure the average precision
                7)  The mAP is calculated by finding Average Precision(AP) for each class and then average over a number of classes

        """

        print("Model evaluation begins !")
        # set it to evaluation mode
        self.model.eval()

        num_elements_validation = len(self.valid_loader)
        prog_bar = tqdm(self.valid_loader, total=num_elements_validation) # we work on the validation set
        # From: https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
        # our model return the predicted boxes in ``[x1, y1, x2, y2]`` format
        # which is the default box_format for MeanAveragePrecision method 
        metric_bbox = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        metric_mask = MeanAveragePrecision(iou_type="segm", class_metrics=True)
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(prog_bar):
                images, targets = data
                # from .....torchvision_tutorial.html
                images =  list(image.to(DEVICE) for image in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                prediction = self.model(images)

                metric_bbox.update(prediction, targets)

                for pred in prediction:
                    pred['masks'] = pred['masks'].squeeze()
                    pred['masks'] = pred['masks'] > 0.5

                metric_mask.update(prediction, targets)

            result_bbox = metric_bbox.compute()
            result_mask = metric_mask.compute()

            print("Model evaluation finished !\n")

            # Write content to the file
            print('result_mask MAP: ')
            print(result_mask)
            print('\nresult_bbox MAP: ')
            print(result_bbox)