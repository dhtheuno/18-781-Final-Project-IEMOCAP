import os
import torch
import math
import torch.nn
from transformers.optimization import Adafactor, AdamW, get_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import ViTEmotion
from datasets import IEMOCAPDataset
import torchvision.transforms as transforms
from tqdm import tqdm

import configargparse

class Trainer: 
    def __init__(self, params: configargparse.Namespace):
        #Initialize some variables
        self.params = params
        self.num_epochs = params.num_epochs
        self.device = params.device
        self.data_dir = params.data_dir
        
        #Initialize Dataset and Dataloader
        trainset = IEMOCAPDataset(
            data_dir=self.data_dir,
            train=True,
            transform=transforms.Compose([
                transforms.PILToTensor()
                #transforms.ToTensor(),
            ])
        )
        testset = IEMOCAPDataset(
            data_dir=self.data_dir,
            train=False,
            transform=transforms.Compose([
                transforms.PILToTensor()
                #transforms.ToTensor(),
            ])
        )
        self.trainloader = DataLoader(
            trainset,
            batch_size = params.batch_size,
            shuffle = True
        )
        self.testloader = DataLoader(
            testset,
            batch_size = params.batch_size,
            shuffle = False
        )

        #Initialize Model
        self.model = ViTEmotion(params)
        if self.device == "cuda":
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            else:
                print("NO GPU AVAILABLE TRAINING IN CPU")
        
        #Initialize Optimizer
        if params.opt == "Adafactor":
            self.opt = Adafactor(
                self.model.parameters(),
                lr = params.lr,
                eps=params.eps,
                weight_decay=params.weight_decay
            )
        else:
            self.opt = AdamW(
                self.model.parameters(),
                lr = params.lr,
                eps=params.eps,
                weight_decay=params.weight_decay
            )
        
        self.epoch = 0

        ## Initialize Stats for Logging
        self.train_stats = {}
        self.val_stats = {}
        self.val_stats["best_acc"] = 0.0
        self.val_stats["best_epoch"] = 0
        self.writer = SummaryWriter(params.tb_dir)

    def train(self):
        for epoch in tqdm(range(self.num_epochs)):
            self.epoch = epoch
            print(f"Starting Training {epoch} Epoch")
            self.reset_stats()

            self.train_epoch()
            self.validate_epoch()
            print(
                "Epoch {}| Training: Loss {:.2f} , Accuracy {:.2f}| Validation: Loss {:.2f} Accuracy {:.2f} WER {:.2f}| ".format(
                    self.epoch, self.train_stats["loss"], self.train_stats["acc"], self.val_stats["loss"],
                    self.val_stats["acc"]))
            self.log_epoch()
            ## Save Models 
            self.save_model()


   
    def train_epoch(self):
        self.model.train()
        for inputs, labels in self.trainloader:
            if self.device == "cuda":
                labels = labels.cuda()
            images = []
            for i in range(len(inputs)):
                images.append(inputs[i,:,:,:])

            self.opt.zero_grad()
            loss, logits = self.model(images, labels)
            loss.backward() 
            
            self.train_stats["nbatches"] += 1

            #Calculate Accuracy
            predicted_labels = torch.argmax(logits,dim=-1)
            batch_length = labels.size()[0]
            acc = torch.sum(predicted_labels == labels)/batch_length
            self.train_stats["acc"] += acc
            self.train_stats["loss"] += loss.item()


            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.grad_clip)
            if math.isnan(grad_norm):
                print("grad norm is nan. Do not update model.")
            else:
                self.opt.step

        self.train_stats["acc"] /= self.train_stats["nbatches"] 
        self.train_stats["loss"] /= self.train_stats["nbatches"]
    
    def validate_epoch(self):
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.testloader:
                if self.device == "cuda":
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                loss, logits = self.model(inputs, labels)
                self.val_stats["nbatches"] += 1

                #Calculate Accuracy
                predicted_labels = torch.argmax(logits,dim=-1)
                batch_length = labels.shape()[0]
                acc = torch.sum(predicted_labels == labels.data())/batch_length
                self.val_stats["acc"] += acc
                self.val_stats["loss"] += loss.item()

            self.val_stats["acc"] /= self.val_stats["nbatches"] 
            self.val_stats["loss"] /= self.val_stats["nbatches"]    
  

    def reset_stats(self):
        """
        Utility function to reset training and validation statistics at the start of each epoch
        """
        self.train_stats["nbatches"] = 0
        self.train_stats["acc"] = 0
        self.train_stats["loss"] = 0
        self.val_stats["nbatches"] = 0
        self.val_stats["acc"] = 0
        self.val_stats["loss"] = 0
    
    def save_model(self):
        """
        Utility function to save the model snapshot after every epoch of training. 
        Saves the model after each epoch as <model-path>/snapshot.ep{}.pth
        Saves the model with highest validation accuracy thus far (and least CER) as <model-path>/model.acc.best
        Updates the best validation accuracy and epoch with the best validation accuracy in validation stats dictionary
        """
        torch.save(
            {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.opt.state_dict(),
                'acc': self.val_stats["acc"],
            }, os.path.join(self.params.model_dir, "snapshot.ep{}.pth".format(self.epoch)))
        if self.val_stats["best_acc"] <= self.val_stats["acc"]:
            self.val_stats["best_acc"] = self.val_stats["acc"]
            self.val_stats["best_epoch"] = self.epoch
            print("Saving model after epoch {}".format(self.epoch))
            torch.save(
                {
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'acc': self.val_stats["acc"],
                }, os.path.join(self.params.model_dir, "model.acc.best"))
        # else:
        #    checkpoint = torch.load(os.path.join(self.params.model_dir,"model.acc.best"))
        #    self.model.load_state_dict(checkpoint["model_state_dict"])

    def log_epoch(self):
        """
        Utility function to write parameters from the Training and Validation Statistics Dictionaries 
        onto Tensorboard at the end of each epoch
        """
        self.writer.add_scalar("training/acc", self.train_stats["acc"], self.epoch)
        self.writer.add_scalar("training/loss", self.train_stats["loss"], self.epoch)
        self.writer.add_scalar("validation/acc", self.val_stats["acc"], self.epoch)
        self.writer.add_scalar("validation/loss", self.val_stats["loss"], self.epoch)
        self.writer.add_scalar("validation/best_acc", self.val_stats["best_acc"], self.epoch)