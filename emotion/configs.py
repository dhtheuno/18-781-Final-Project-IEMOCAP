from torch import optim, nn
import models
from datasets import get_loader
import torchvision.transforms as transforms

class VGG16Config:
    def __init__(self):
        self.model = models.VGG16Emotion()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.n_epochs = 50
        self.train_loader = get_loader(split='train', features='spectrogram_v1', batch_size=32)
        self.val_loader = get_loader(split='val', features='spectrogram_v1', batch_size=32)
        self.test_loader = get_loader(split='test', features='spectrogram_v1', batch_size=32)

class VGG16FCConfig:
    def __init__(self):
        self.model = models.VGG16FCEmotion()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.n_epochs = 50
        self.train_loader = get_loader(split='train', features='spectrogram_v1', batch_size=32)
        self.val_loader = get_loader(split='val', features='spectrogram_v1', batch_size=32)
        self.test_loader = get_loader(split='test', features='spectrogram_v1', batch_size=32)

class VGG16BWConfig:
    def __init__(self):
        self.model = models.VGG16Emotion()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.n_epochs = 50
        self.train_loader = get_loader(split='train', features='spectrogram_v2', batch_size=32)
        self.val_loader = get_loader(split='val', features='spectrogram_v2', batch_size=32)
        self.test_loader = get_loader(split='test', features='spectrogram_v2', batch_size=32)

class VGG16FrozenConfig:
    def __init__(self):
        self.model = models.VGG16Emotion(frozen=True)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.n_epochs = 50
        self.train_loader = get_loader(split='train', features='spectrogram_v1', batch_size=32)
        self.val_loader = get_loader(split='val', features='spectrogram_v1', batch_size=32)
        self.test_loader = get_loader(split='test', features='spectrogram_v1', batch_size=32)

class VGG16AugmentedConfig:
    def __init__(self):
        self.model = models.VGG16Emotion()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.n_epochs = 100
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.1)),
        ])
        self.train_loader = get_loader(split='train', features='spectrogram_v1', batch_size=32, transform=transform)
        self.val_loader = get_loader(split='val', features='spectrogram_v1', batch_size=32)
        self.test_loader = get_loader(split='test', features='spectrogram_v1', batch_size=32)


class VGG19Config:
    def __init__(self):
        self.model = models.VGG19Emotion()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.n_epochs = 50
        self.train_loader = get_loader(split='train', features='spectrogram_v1', batch_size=32)
        self.val_loader = get_loader(split='val', features='spectrogram_v1', batch_size=32)
        self.test_loader = get_loader(split='test', features='spectrogram_v1', batch_size=32)

class InceptionV3Config:
    def __init__(self):
        self.model = models.InceptionV3Emotion()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.n_epochs = 50
        self.train_loader = get_loader(split='train', features='spectrogram_v1', batch_size=32)
        self.val_loader = get_loader(split='val', features='spectrogram_v1', batch_size=32)
        self.test_loader = get_loader(split='test', features='spectrogram_v1', batch_size=32)

class Resnet18Config:
    def __init__(self):
        self.model = models.Resnet18Emotion()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.n_epochs = 50
        self.train_loader = get_loader(split='train', features='spectrogram_v1', batch_size=32)
        self.val_loader = get_loader(split='val', features='spectrogram_v1', batch_size=32)
        self.test_loader = get_loader(split='test', features='spectrogram_v1', batch_size=32)

class Resnet152Config:
    def __init__(self):
        self.model = models.Resnet152Emotion()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-5, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.n_epochs = 50
        self.train_loader = get_loader(split='train', features='spectrogram_v1', batch_size=32)
        self.val_loader = get_loader(split='val', features='spectrogram_v1', batch_size=32)
        self.test_loader = get_loader(split='test', features='spectrogram_v1', batch_size=32)

class AlexnetConfig:
    def __init__(self):
        self.model = models.AlexnetEmotion()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.n_epochs = 25
        self.train_loader = get_loader(split='train', features='spectrogram_v1', batch_size=32)
        self.val_loader = get_loader(split='val', features='spectrogram_v1', batch_size=32)
        self.test_loader = get_loader(split='test', features='spectrogram_v1', batch_size=32)


def get_config(tag):
    # add new configs to this dictionary
    config_map = {
        'vgg16': VGG16Config,
        'vgg16_fc': VGG16FCConfig,
        'vgg16_bw': VGG16BWConfig,
        'vgg16_frozen': VGG16FrozenConfig,
        'vgg16_augmented': VGG16AugmentedConfig,
        'resnet18': Resnet18Config,
        'resnet152': Resnet152Config,
        'alexnet': AlexnetConfig,
        'vgg19': VGG19Config,
        'inceptionv3': InceptionV3Config,
    }

    return config_map[tag]()

if __name__ == '__main__':
    print(get_config('vgg16'))
