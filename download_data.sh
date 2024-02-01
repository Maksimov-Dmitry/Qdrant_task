#!/bin/bash

mkdir -p data/raw

for i in 0 1; do
    wget "https://storage.googleapis.com/ads-dataset/subfolder-$i.zip" &&
    unzip "subfolder-$i.zip" -d data/raw &&
    rm "subfolder-$i.zip"
done
