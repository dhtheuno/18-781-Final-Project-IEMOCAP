import torch
import time
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.models as models
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter

import pandas as pd
from skimage import io

from datasets import IEMOCAPDataset

def train(model, criterion, optimizer, scheduler, trainloader, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        pred_counter = Counter()
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in tqdm(trainloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    pred_counter.update(preds.data.cpu().numpy())
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / len(trainloader)
            epoch_acc = running_corrects.double() / len(trainloader.dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('prediction counts', pred_counter)
            # deep copy the model
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model

def test(model, testloader):
    pred_counter = Counter()
    model.eval()
    test_loss = 0
    correct = 0
    test_losses = []
    with torch.no_grad():
        for data, target in tqdm(testloader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = torch.max(output.data, 1)[1]
            pred_counter.update(pred.data.cpu().numpy())
            correct += (pred == target).sum()
    test_loss /= len(testloader.dataset)
    test_losses.append(test_loss)
    print('prediction counts', pred_counter)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

if __name__ == '__main__':
    data_dir = './data/iemocap'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset = IEMOCAPDataset(
        data_dir=data_dir,
        split='train',
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    testset = IEMOCAPDataset(
        data_dir=data_dir,
        split='val',
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    trainloader = DataLoader(
        trainset,
        batch_size = 32,
        shuffle = True
    )

    testloader = DataLoader(
        testset,
        batch_size = 8,
        shuffle = True
    )
    num_classes = 4
    model_vgg = models.vgg16(pretrained=True)
    # model_vgg.features[0] = nn.Conv2d(3, 64, 3, 1, 1)
    model_vgg.classifier[-1] = nn.Linear(4096, 1000)
    model_vgg.classifier.add_module('7', nn.ReLU())
    model_vgg.classifier.add_module('8', nn.Dropout(p=0.5, inplace=False))
    # model_vgg.classifier.add_module('9', nn.ReLU())
    model_vgg.classifier.add_module('10', nn.Linear(1000, num_classes))
    # model_vgg.classifier.add_module('10', nn.LogSoftmax(dim=1))
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_vgg.parameters(), lr=1e-4, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)
    model_vgg.to(device)

    model_vgg = train(model_vgg, criterion, optimizer_ft, exp_lr_scheduler, trainloader, num_epochs=50)
    test(model_vgg, testloader)
