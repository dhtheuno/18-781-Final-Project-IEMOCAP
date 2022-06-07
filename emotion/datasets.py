import pandas as pd
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import Dataset, DataLoader
import numpy as np


class IEMOCAPDataset(Dataset):

    def __init__(self, data_dir='data/iemocap', split='train', spectrogram_dir='spectrogram_v1', transform=None):

        self.df = pd.read_csv(f'{data_dir}/{split}.csv')
        self.data_dir = data_dir
        self.spectrogram_dir = spectrogram_dir
        self.transform = transform

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
        image = io.imread(f'{self.data_dir}/{self.spectrogram_dir}/{file_id}.png')
        image = image[:,:,0:3] # remove alpha channel
        label = self.label_map[self.df['emotion_label'][idx]]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_loader(data_dir='data/iemocap', features='spectrogram_v1', batch_size=32, split='train',
               transform=transforms.Compose([transforms.ToTensor()])):

    dataset = IEMOCAPDataset(
        data_dir=data_dir,
        spectrogram_dir=features,
        split=split,
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=split=='train')
    return loader


def prepare_splits(data_dir):
    df = pd.read_csv(f'{data_dir}/labels.csv')

    # remove extra labels
    for l in ['xxx', 'fru', 'sur', 'fea', 'oth', 'dis']:
        df = df[df.emotion_label != l]
    df.loc[df.emotion_label == 'exc', 'emotion_label'] = 'hap' # combine happy and excited labels

    # train/test split
    train_df = df[df.file_id < 'Ses05'].reset_index(drop=True)
    valtest_df = df[df.file_id > 'Ses05'].reset_index(drop=True)

    np.random.seed(42)
    valtest_indices = np.random.permutation(valtest_df.index)

    val_size = 400
    val_df = valtest_df.loc[valtest_indices[:val_size]].reset_index(drop=True)
    test_df = valtest_df.loc[valtest_indices[val_size:]].reset_index(drop=True)

    train_df.to_csv(f'{data_dir}/train.csv')
    val_df.to_csv(f'{data_dir}/val.csv')
    test_df.to_csv(f'{data_dir}/test.csv')

if __name__ == '__main__':
    prepare_splits('data/iemocap')
    trainset = IEMOCAPDataset(split='train')
    print('train', len(trainset))
    valset = IEMOCAPDataset(split='val')
    print('val', len(valset))
    testset = IEMOCAPDataset(split='test')
    print('test', len(testset))
