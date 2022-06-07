import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
class IEMOCAPDataset(Dataset):

    def __init__(self, data_dir='', csv_fname='labels.csv', spectrogram_dir='spectrogram_v1', train=True, transform=None):

        self.df = pd.read_csv(f'{data_dir}/{csv_fname}')
        self.data_dir = data_dir
        self.spectrogram_dir = spectrogram_dir
        self.transform = transform

        # remove extra labels
        for l in ['xxx', 'fru', 'sur', 'fea', 'oth', 'dis']:
            self.df = self.df[self.df.emotion_label != l]
        self.df.loc[self.df.emotion_label == 'exc', 'emotion_label'] = 'hap' # combine happy and excited labels

        # train/test split
        if train:
            self.df = self.df[self.df.file_id < 'Ses05']
        else:
            self.df = self.df[self.df.file_id > 'Ses05']

        self.df = self.df.reset_index(drop=True)

        self.label_map = {
            'hap': 0,
            'sad': 1,
            'ang': 2,
            'neu': 3,
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_id = self.df['file_id'][idx]

        #image = io.imread(f'{self.data_dir}/{self.spectrogram_dir}/{file_id}.png')
        #image = image[:,:,0:3] # remove alpha channel
        image = Image.open(f'{self.data_dir}/{self.spectrogram_dir}/{file_id}.png')
        image = image.convert("RGB")
        label = self.label_map[self.df['emotion_label'][idx]]

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    trainset = IEMOCAPDataset(
        data_dir='./data/iemocap',
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    print('train length', len(trainset))
    for images, labels in trainloader:
        print('batch shape', images.shape)
        print('label', labels[0].item())
        # plt.imshow(images[0].permute(1, 2, 0))
