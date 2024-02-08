import torch
import torch.optim as optim

import torchvision
import torch
import torchaudio

from tqdm import tqdm
from model.mask_rcnn import MaskRCNN

NUM_EPOCHS = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#torchvision-0.17.0+cu118-cp311-cp311-win_amd64.whl
""" print(torch.__version__)
print(torchvision.__version__)
print(torchaudio.__version__) """

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
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-5)

    def train(self) :
        self.model.train()

        for epoch in range(NUM_EPOCHS):
            prog_bar = tqdm(self.train_loader, total=len(self.train_loader))
            
            for i, data in enumerate(prog_bar):
                images, targets = data
                # from .....torchvision_tutorial.html
                images =  list(image.to(DEVICE) for image in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                self.optimizer.zero_grad()

                output = self.model(images, targets)

                print(output)

                #print(type(output))
                # capire il criterion da utilizzare
                #loss = criterion(outputs, labels)
                #loss.backward()
                #optimizer.step()
                
                #print(output)

        print("Training Finished !")

    def validate(self) :
        return
    
    def test(self) :
        return