import pandas as pd
import numpy as np
import torchaudio
from tqdm import tqdm
from collections import Counter

def get_metadata(file_id):
    DATA_DIR = './data/iemocap'
    filename = f'{DATA_DIR}/wav/{file_id[:-5]}/{file_id}.wav'
    return torchaudio.info(filename)

if __name__ == '__main__':

    df = pd.read_csv('./data/iemocap/labels.csv')

    # remove extra labels
    for l in ['xxx', 'fru', 'sur', 'fea', 'oth', 'dis']:
        df = df[df.emotion_label != l]
    df.loc[df.emotion_label == 'exc', 'emotion_label'] = 'hap'

    session_counts = Counter()
    label_counts = Counter()
    total_count = 0

    sessions = [i[:5] for i in df['file_id']]
    session_counts.update(sessions)
    print('sessions:')
    for key in sorted(session_counts.keys()):
        print(f'{key}: {session_counts[key]}')

    labels = [i for i in df['emotion_label']]
    label_counts.update(labels)
    print('\nlabels:')
    print(sorted(((v, k) for k, v in label_counts.items()), reverse=True))


    for i, row in df.iterrows():
        total_count += 1

    print()
    print('Train: ', sum([session_counts[f'Ses0{i}'] for i in range(1, 5)]))
    print('Test: ', session_counts['Ses05'])
    print('Total: ', total_count)

    lengths = {}

    for file_id in tqdm(list(df['file_id'])):
        metadata = get_metadata(file_id)
        lengths[file_id] = metadata.num_frames

    print('max', max(lengths.items(), key = lambda k: k[1]))
    print('min', min(lengths.items(), key = lambda k: k[1]))
