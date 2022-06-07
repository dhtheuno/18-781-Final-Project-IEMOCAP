import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

# This script generates a spectrogram (224 x 224 px) for each input utterance

DATA_DIR = './data/iemocap'
OUTPUT_DIR = './data/iemocap/spectrogram_v1'

def get_waveform(file_id):
    filename = f'{DATA_DIR}/wav/{file_id[:-5]}/{file_id}.wav'
    return torchaudio.load(filename)

def get_metadata(file_id):
    filename = f'{DATA_DIR}/wav/{file_id[:-5]}/{file_id}.wav'
    return torchaudio.info(filename)

def get_spectrogram_v1(file_id):
    # v1 uses n_fft=1024, n_mels=224, concatenates if too short and truncates if too long

    waveform, fs = get_waveform(file_id)

    transforms = torchaudio.transforms.MelSpectrogram(n_fft=1024, n_mels=224)
    spec = transforms(waveform)
    log_spec = torch.log(spec)

    time_shape = spec.shape[2]
    n_copies = math.floor(224 / time_shape)
    remainder = 224 % time_shape

    # should we concatenate like this? other options are to stretch, just add silence, or reduce frequency resolution (probably bad)
    image = torch.cat([log_spec] * n_copies + [log_spec[:,:,:remainder]], dim=2)
    assert image.shape[1] == 224 and image.shape[2] == 224

    return image[0]

if __name__ == '__main__':

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(f'{DATA_DIR}/labels.csv')
    file_ids = list(df['file_id'])

    min_value = np.inf
    max_value = -np.inf

    for file_id in tqdm(file_ids):
        image = get_spectrogram_v1(file_id)
        min_value = min(min_value, torch.min(image))
        max_value = max(max_value, torch.max(image))
        plt.imsave(f'{OUTPUT_DIR}/{file_id}.png', image)

    print('min', min_value)
    print('max', max_value)

