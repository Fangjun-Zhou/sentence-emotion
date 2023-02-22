# Add word2vec to the python path.
import sys
sys.path.append("external/word2vec")

import os
import argparse
import yaml
import zipfile

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from external.word2vec.train import train
from external.word2vec.utils.helper import (
    get_model_class,
    get_optimizer_class,
    get_lr_scheduler,
    save_vocab,
    load_vocab,
)

CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Read the emotion text data.
emotion_df = pd.read_csv("data/text-emotion.zip")

# Dataset for twitter financial news text.
class EmotionTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size = -1):
        self.emotion_text = df
        # Shuffle and take a subset of the data.
        if size > 0:
            self.emotion_text = self.emotion_text.sample(frac=1).reset_index(drop=True)
            self.emotion_text = self.emotion_text[:size]
        
    def __len__(self):
        return len(self.emotion_text)
    
    def __getitem__(self, idx):
        return self.emotion_text.iloc[idx, 0]

# Read in the datset.
emotion_dataset = EmotionTextDataset(emotion_df, size = config["dataset_size"])

if (config["pre_trained_vocab_path"]):
    vocab: Vocab = load_vocab(config["pre_trained_vocab_path"])
    vocab_size = len(vocab.get_stoi())
    print(f"Pretrained vocab size: {vocab_size}")
else:
    vocab = None

# Get the english tokenizer.
tokenizer = get_tokenizer("basic_english", language="en")
# Build the extended vocab based on dataset.
extend_vocab = build_vocab_from_iterator(
    map(tokenizer, emotion_dataset),
    specials=["<unk>"],
    min_freq=config["vocab_min_word_frequency"]
)
extend_vocab.set_default_index(extend_vocab["<unk>"])

if vocab:
    new_token = []
    for word in extend_vocab.get_stoi():
        if not word in vocab:
            new_token.append(word)
    # Add all new tokens to the vocab.
    for token in new_token:
        vocab.append_token(token)
    print(f"{len(new_token)} new tokens added to the vocab.")
    vocab_size = len(vocab.get_stoi())
    print(f"Extended vocab size: {vocab_size}")
else:
    vocab = extend_vocab
    vocab_size = len(vocab.get_stoi())
    print(f"Extended vocab size: {vocab_size}")

# Get the pretrained model.
if config["pre_trained_model_path"]:
    pretrained_model = torch.load(config["pre_trained_model_path"], map_location=torch.device("cpu"))
else:
    pretrained_model = None

if pretrained_model:
    train(
        config=config,
        data_iter=emotion_dataset,
        vocab=vocab,
        transfer_model=pretrained_model
    )
else:
    train(
        config=config,
        data_iter=emotion_dataset,
        vocab=vocab
    )