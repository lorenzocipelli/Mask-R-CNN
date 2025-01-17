import torch
import argparse

from utils.utils import collate_fn

from torch.utils.data import DataLoader
from utils.modanet import ModaNetDataset
from utils.utils import transforms_pipeline
from engine import Engine

# print('getcwd: ', os.getcwd()) # working directory

FIRST_STOP = 32728 # 70 %
SECOND_STOP = 7013 # 15 %

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default="", help='name of the model to be saved/loaded')
    parser.add_argument('--annotations_file', type=str, default="modanet2018_instances_train_fix.json", help='name of the annotations file')

    parser.add_argument('--epochs', type=int, default=4, help='number of epochs in training')
    parser.add_argument('--batch_size', type=int, default=4, help='number of elements in batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in data loader')

    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--opt', type=str, default='SGD', choices=['SGD', 'Adam'], help = 'training optimizer')

    parser.add_argument('--dataset_path', type=str, default='./dataset', help='path were to save/get the dataset')
    parser.add_argument('--saving_path', type=str, default='./model/checkpoints/', help='path where to save the trained model')

    parser.add_argument('--resume', action='store_true', help='load the model from checkpoint (true or false)')
    parser.add_argument('--resume_name', type=str,  default='', help='checkpoint model name')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test','hyper_tuning'], help = 'net mode (train or test or hyperparamter tuning)')
    parser.add_argument('--pretrained', action='store_true', help='load pretrained coco weights')
    parser.add_argument('--use_amp', action='store_true', help='use Automatic Mixed Precision (AMP) to speed-up training')
    parser.add_argument('--version', type=str, default='V1', choices=['V1', 'V2'], help = 'maskrcnn version V1 or V2')
    parser.add_argument('--use_accessory', action='store_true', help='add new feature as accessory')

    parser.add_argument('--custom_loss', action='store_true', help='implement an addictive custom loss')

    return parser.parse_args()

def get_subsets(dataset) :
    # total -> 46754 elements (100 %)
    # train -> 32728 elements (70 %)
    # validation -> 7013 elements (15 %)
    # test -> 7013 elements (15 %)

    idxs = torch.randperm(len(dataset)).tolist()
    # modificare di nuovo, messo a mille per ridurre l'uso di memoria durante lo sviluppo
    train_list, valid_list, test_list = idxs[:FIRST_STOP], idxs[FIRST_STOP:FIRST_STOP+SECOND_STOP], idxs[-SECOND_STOP:]
    train = torch.utils.data.Subset(dataset, train_list)
    valid = torch.utils.data.Subset(dataset, valid_list)
    test = torch.utils.data.Subset(dataset, test_list)

    return train, valid, test

def main(args) :
    modanet = ModaNetDataset(args, transforms_pipeline())

    train_modanet, val_modanet, test_modanet = get_subsets(modanet)
    # collate_fn to put all the data inside the DataLoader "queue" inside a single batch element. It enables -> for batch in train_loader
    train_loader = DataLoader(train_modanet, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    valid_loader = DataLoader(val_modanet, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_modanet, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    if args.mode == "hyper_tuning" :
        hyperparams = {
            "learning_rate" : [0.005, 0.0005],
            "batch_size" : [16,32],
            "epochs" : [8,12]
        }

        for lr in hyperparams['learning_rate'] :
            for bs in hyperparams['batch_size'] :
                for ep in hyperparams['epochs'] :
                    run_name = "lr_" + str(lr).replace(".","-") + "_bs_" + str(bs) + "_ep_" + str(ep)
                    args.model_name = run_name
                    args.lr = lr
                    args.batch_size = bs
                    args.epochs = ep

                    engine = Engine(train_loader, valid_loader, test_loader, args)
                    engine.train()     

                    print("Training of " + run_name + " finished !")

                    del engine       

        print("Hyperparameters fine-tuning finished !")
        
    else : # training or testing
        engine = Engine(train_loader, valid_loader, test_loader, args)

        if args.mode == "train" :
            engine.train()
        elif args.mode == "test" :
            engine.test()

if __name__ == "__main__" :
    args = get_args()
    main(args)