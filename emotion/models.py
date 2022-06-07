import torch
from torch import nn
from torchvision import models

class VGG16Emotion(torch.nn.Module):
    def __init__(self, pretrained=True, frozen=False, num_classes=4):
        super(VGG16Emotion, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier[-1] = nn.Linear(4096, 1000)
        self.model.classifier.add_module('7', nn.ReLU())
        self.model.classifier.add_module('8', nn.Dropout(p=0.5, inplace=False))
        self.model.classifier.add_module('10', nn.Linear(1000, num_classes))
        print(self.model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

class VGG16FCEmotion(torch.nn.Module):
    def __init__(self, pretrained=True, num_classes=4):
        super(VGG16FCEmotion, self).__init__()
        self.model = models.vgg16_bn(pretrained=True)
        self.model.classifier[-1] = nn.Sequential(nn.Linear(4096, num_classes))
        for param in self.model.classifier[-1].parameters():
            param.requires_grad = True

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

class VGG19Emotion(torch.nn.Module):
    def __init__(self, pretrained=True, num_classes=4):
        super(VGG19Emotion, self).__init__()
        self.model = models.vgg19(pretrained=pretrained)
        self.model.classifier[-1] = nn.Linear(4096, 1000)
        self.model.classifier.add_module('7', nn.ReLU())
        self.model.classifier.add_module('8', nn.Dropout(p=0.5, inplace=False))
        self.model.classifier.add_module('10', nn.Linear(1000, num_classes))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

class InceptionV3Emotion(torch.nn.Module):
    def __init__(self, pretrained=True, num_classes=4):
        super(InceptionV3Emotion, self).__init__()
        self.model = models.inception_v3(pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1000, num_classes)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

class Resnet18Emotion(torch.nn.Module):
    def __init__(self, pretrained=True, num_classes=4):
        super(Resnet18Emotion, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1000, num_classes)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

class Resnet152Emotion(torch.nn.Module):
    def __init__(self, pretrained=True, num_classes=4):
        super(Resnet152Emotion, self).__init__()
        self.model = models.resnet152(pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1000, num_classes)
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

class AlexnetEmotion(torch.nn.Module):
    def __init__(self, pretrained=True, num_classes=4):
        super(AlexnetEmotion, self).__init__()
        self.model = models.alexnet(pretrained=pretrained)
        num_ftrs = self.model.classifier[1].in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1000, num_classes)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = VGG16Emotion()
    print(model)


