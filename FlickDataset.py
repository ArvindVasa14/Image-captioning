import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer
from PIL import Image
import os


class FlickDataset(Dataset):
    def __init__(self, root_dir, caption_file, images_path, transform=None, freq_threshold= 5):
        self.root_dir= root_dir
        self.caption_file= caption_file
        self.images_path= images_path
        self.freq_threshold= freq_threshold
        self.transform= transform

        self.df= pd.read_csv(self.caption_file)
        self.images= self.df["image"]
        self.captions= self.df["caption"]

        self.tokenizer= get_tokenizer("basic_english")
        self.vocab= build_vocab_from_iterator(
            self.yield_tokens(self.captions.tolist()),
            specials=["<UNK>", "<SOS>", "<EOS>", "<PAD>"]
        )
        self.vocab.set_default_index(self.vocab["<UNK>"])

        self.text_pipeline= lambda caption: self.vocab(self.tokenizer(caption))

    # generator to return one after the other upon request
    def yield_tokens(self, captions):
        for caption in captions:
            return self.tokenizer(caption)

    # len
    def __len__(self):
        return len(self.df)

    #getitem
    def __getitem__(self, index):
        # reading image
        img_id= self.images[index]
        img_path= os.path.join(self.root_dir, img_id)
        img= Image.open(img_path)

        # reading caption
        caption= self.captions[index]

        if self.transform:
            img= self.transform(img)

        numericalized_caption= self.vocab["<SOS>"]
        numericalized_caption+= self.text_pipeline(caption)
        numericalized_caption+= self.vocab["<EOS>"]

