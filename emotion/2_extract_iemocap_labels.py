# script to extract all labels from IEMOCAP dataset into a single Pandas dataframe
# adapted from https://github.com/Aditya3107/IEMOCAP_EMOTION_Recognition/blob/master/1_extract_emotion_labels.ipynb

import re
import os
import pandas as pd

# This script expects this data directory to be already organized (e.g. after running 1_organize_iemocap.sh)
IEMOCAP_DIR = './data/iemocap/labels_raw'
OUTPUT_FILE = './data/iemocap/labels.csv'

if __name__ == '__main__':
    data_dict = {'file_id': [], 'emotion_label': [], 'val': [], 'act': [], 'dom': []}
    label_files = sorted(os.listdir(IEMOCAP_DIR))
    for label_file in label_files:
        with open(os.path.join(IEMOCAP_DIR, label_file)) as lf:
            for line in lf:
                if line[0] == '[': # this is a relevant info line
                    start_end_time, file_id, emotion_label, val_act_dom = line.strip().split('\t')
                    # start_time, end_time = start_end_time[1:-1].split('-') # we ignore start/end time, uncomment if necessary
                    val, act, dom = val_act_dom[1:-1].split(',')
                    val, act, dom = float(val), float(act), float(dom)
                    data_dict['file_id'].append(file_id)
                    data_dict['emotion_label'].append(emotion_label)
                    data_dict['val'].append(val)
                    data_dict['act'].append(act)
                    data_dict['dom'].append(dom)

    iemocap_df = pd.DataFrame(data_dict)
    # print(iemocap_df.tail())

    iemocap_df.to_csv(OUTPUT_FILE, index=False)








