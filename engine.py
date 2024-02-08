import os
import sys
import math
import torch
import torch.optim as optim

from tqdm import tqdm
from model.mask_rcnn import MaskRCNN

NUM_EPOCHS = 4
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

    def __init__(self, train_load, valid_load, test_load) :
        self.model = MaskRCNN(num_classes=14).to(DEVICE)
        self.train_loader = train_load
        self.valid_loader = valid_load
        self.test_loader = test_load
        # The call to model.parameters()
        # in the SGD constructor will contain the learnable parameters (defined 
        # with torch.nn.Parameter) which are members of the model
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-5)

    def save_model(self, epoch) :
        # if you want to save the model
        checkpoint_name = "epoch" + str(epoch) + "_" + "mask_r_cnn"
        check_path = os.path.join("./model/", checkpoint_name)
        torch.save(self.model.state_dict(), check_path)
        print("Model saved!")

    def load_model(self):
        # function to load the model
        check_path = os.path.join("./model/", "epoch" + str(0) + "_" + "mask_r_cnn")
        self.net.load_state_dict(torch.load(check_path, map_location=torch.device(DEVICE)))
        print("Model loaded!", flush=True)

    def train(self) :
        self.model.train()
        set_debug_apis(state=False)

        for epoch in range(NUM_EPOCHS):
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
                
                # from https://github.com/pytorch/vision/blob/main/references/detection/engine.py
                # If you have multiple losses you can sum them and then call backwards once
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")
                    print(loss_dict)
                    sys.exit(1)

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
        return