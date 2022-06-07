#!/bin/bash

# script to organize audio and label files of the IEMOCAP dataset,
# and delete all unneccesary files (mocap, videos, etc)

IEMOCAP_DIR=~/local/iemocap/IEMOCAP_full_release
OUTPUT_DIR=./data/iemocap

mkdir -p $OUTPUT_DIR

for i in $(seq 1 5); do
    echo "copying wav files for session $i"
    wav_dir="$IEMOCAP_DIR/Session$i/sentences/wav/"
    rsync -a $wav_dir "$OUTPUT_DIR/wav"

    echo "copying labels for session $i"
    label_dir="$IEMOCAP_DIR/Session$i/dialog/EmoEvaluation/*.txt"
    rsync -a $label_dir "$OUTPUT_DIR/labels_raw"
done

echo "done!"

