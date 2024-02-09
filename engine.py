import os
import sys
import math
import torch
import numpy as np
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from model.mask_rcnn import MaskRCNN

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.model = MaskRCNN(num_classes=14, args=args).to(DEVICE)
        self.epochs = args.epochs
        self.train_loader = train_load
        self.valid_loader = valid_load
        self.test_loader = test_load
        self.model_name = args.model_name
        self.check_path = args.saving_path
        self.use_amp = args.use_amp
        # The call to model.parameters()
        # in the SGD constructor will contain the learnable parameters (defined 
        # with torch.nn.Parameter) which are members of the model
        if args.opt == "SGD":
                self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)
        elif args.opt == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    def save_model(self, epoch) :
        # if you want to save the model
        checkpoint_name = str(self.model_name) + "_epoch" + str(epoch) + "_" + str(2999) + "_mask_r_cnn"
        check_path = os.path.join(self.check_path, checkpoint_name)
        torch.save(self.model.state_dict(), check_path)
        print("Model saved!")

    def load_model(self):
        # function to load the model
        check_path = os.path.join(self.check_path, str(self.model_name) + "_epoch" + str(0) + "_" + str(2999) + "_mask_r_cnn")
        self.model.load_state_dict(torch.load(check_path, map_location=torch.device(DEVICE)))
        print("Model loaded!", flush=True)

    def train(self) :
        self.model.train()

        set_debug_apis(state=False)
        if self.use_amp :
            # initialize gradient scaler
            scaler = torch.cuda.amp.GradScaler() 

        for epoch in range(self.epochs):
            prog_bar = tqdm(self.train_loader, total=len(self.train_loader))
            """ 
                Our model compute the losses for every head, and return them in a dictionary 
                like the following :
                {'loss_classifier': tensor(2.8990, device='cuda:0', grad_fn=<NllLossBackward0>),
                 'loss_box_reg': tensor(0.3307, device='cuda:0', grad_fn=<DivBackward0>),
                 'loss_mask': tensor(1.2220, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>),
                 'loss_objectness': tensor(0.3347, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>),
                 'loss_rpn_box_reg': tensor(0.0507, device='cuda:0', grad_fn=<DivBackward0>) } """
            
            loss_dict = {
                'loss_classifier': 0,
                'loss_box_reg': 0,
                'loss_mask': 0,
                'loss_objectness': 0,
                'loss_rpn_box_reg': 0
            }

            running_loss = 0.0
            
            for i, data in enumerate(prog_bar):
                images, targets = data
                # from .....torchvision_tutorial.html
                images =  list(image.to(DEVICE) for image in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                if self.use_amp :
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

                if self.use_amp :
                    scaler.scale(losses).backward() # compute gradients
                    scaler.step(self.optimizer) # update parameters of the model
                    scaler.update()
                
                else :
                    losses.backward() # compute gradients
                    self.optimizer.step() # update parameters of the model

                # print statistics
                running_loss += loss_value
                if i % 200 == 199: # print every 200 mini-batches
                    print(f'[it: {i + 1}] loss: {running_loss / 200:.3f}')
                    running_loss = 0.0 


        print("Training Finished !")

    def validate(self) :
        return
    
    def test(self) :
        self.load_model()
         # set it to evaluation mode
        self.model.eval()
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            prog_bar = tqdm(self.test_loader, total=len(self.test_loader))
            for i, data in enumerate(prog_bar):
                images, targets = data

                predictions = self.model([images[0]])
                pred = predictions[0]

                image = (255.0 * (images[0] - images[0].min()) / (images[0].max() - images[0].min())).to(torch.uint8)
                image = image[:3, ...]

                preds = [(f"{label}: {score:.3f}", boxes, masks) for label, score, boxes, masks in zip(pred["labels"], pred["scores"], pred["boxes"], pred["masks"]) if score >= 0.07]

                pred_labels =[t[0] for t in preds]
                pred_boxes = torch.as_tensor([np.array(t[1]) for t in preds])
                pred_boxes = pred_boxes.long()
                pred_masks = torch.tensor([np.array(t[2]) for t in preds])
                pred_masks = (pred_masks > 0.5).squeeze(1)
                
                """ pred_labels = [f"{label}: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
                pred_boxes = pred["boxes"].long()
                masks = (pred["masks"] > 0.7).squeeze(1) # valore di sicurezza """

                output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")
                output_image = draw_segmentation_masks(output_image, pred_masks, alpha=0.5, colors="blue")

                plt.figure(figsize=(16, 16))
                plt.imshow(output_image.permute(1, 2, 0))
                plt.show()

                """ img = draw_bounding_boxes(images[0].type(torch.uint8), prediction[0]["boxes"].clone().detach(), width=3)
                img = torchvision.transforms.ToPILImage()(img)
                img.show() """



        return