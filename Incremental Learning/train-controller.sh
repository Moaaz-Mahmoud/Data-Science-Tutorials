#!/bin/bash

GREEN='\033[0;32m'
NC='\033[0m'


for i in {1..3}; do
    echo -e "Training batch ${GREEN}$i${NC}..."

    while [[ $(cat train-status.txt) -ne 1 ]]; do
        sleep 1
    done

    filename="dataset-split/batch-$(printf '%03d' $i).npy"
    echo -e $filename
    python3 train-cnn-cont.py "$filename"
done

echo -e "Training completed!"