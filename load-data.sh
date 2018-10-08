#!/usr/bin/env bash

username=discmonkey
key=62e759932ad9d358e7db9601892f6502

#KAGGLE_USERNAME=${username} KAGGLE_KEY=${key} kaggle datasets download -d uciml/mushroom-classification -p datasets/raw
KAGGLE_USERNAME=${username} KAGGLE_KEY=${key} kaggle datasets download -d dansbecker/nba-shot-logs -p datasets/raw

pushd datasets/raw
unzip nba-shot-logs.zip

popd